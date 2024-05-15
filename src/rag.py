### Import libraries ###
import logging
import os
import shutil
import tempfile
import time
from typing import List
from urllib.request import urlretrieve

import numpy as np
import requests
import streamlit as st
from dotenv import load_dotenv
from fastapi import APIRouter, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse
from langchain.chains import (
    ConversationalRetrievalChain,
    RetrievalQA,
    create_history_aware_retriever,
    create_retrieval_chain,
)
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFDirectoryLoader,
    PyPDFLoader,
    WebBaseLoader,
)
from langchain_community.embeddings import (
    HuggingFaceBgeEmbeddings,
    HuggingFaceEmbeddings,
)
from langchain_community.llms import CTransformers, HuggingFacePipeline
from langchain_community.vectorstores import FAISS
from langchain_core.messages import SystemMessage
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
    PromptTemplate,
)
from langchain_core.runnables import RunnablePassthrough
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from src import model

### Initialization ###
logger = logging.getLogger(__name__)

router = APIRouter()

load_dotenv()

vdb_path = "vectordb/faiss"

embedding_model = HuggingFaceEmbeddings(model_name="Snowflake/snowflake-arctic-embed-s")


### Functions ###
def load_and_split_documents(
    doc_list: list,
    splitter=RecursiveCharacterTextSplitter,
    chunk_size=700,
    chunk_overlap=50,
):
    docs_before_split = []

    for doc_path in doc_list:
        loader = PyPDFLoader(doc_path)
        loaded_docs = loader.load()
        logger.info(f"Loaded {len(loaded_docs)} docs from {doc_path}")
        docs_before_split.extend(loaded_docs)

    # can add text cleaning steps too

    text_splitter = splitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    docs_after_split = text_splitter.split_documents(docs_before_split)
    return docs_after_split


def get_retriever(vdb, search_type="similarity", search_kwargs={"k": 3}):
    # retriever = vdb.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    retriever = vdb.as_retriever(score_threshold=0.6)
    return retriever


def get_response_chain(input, prompt_template, llm, retriever, chat_history):
    full_prompt = ChatPromptTemplate.from_template(prompt_template)
    logger.info(f"Full prompt: {full_prompt}")

    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, full_prompt
    )
    # logger.info(history_aware_retriever)

    document_chain = create_stuff_documents_chain(llm, full_prompt)
    # logger.info(document_chain)

    retrieval_chain = create_retrieval_chain(history_aware_retriever, document_chain)
    logger.info(f"Retrieval chain: {retrieval_chain}")

    # response = retrieval_chain.invoke({"input": input, "chat_history": ""})
    response = retrieval_chain.invoke({"input": input, "chat_history": chat_history})

    start_marker = "AI:"
    start_index = response["answer"].rfind(start_marker) + len(start_marker)
    filtered_response = response["answer"][start_index:].strip()
    logger.info(f"Response: {filtered_response}")
    return filtered_response


### APIs ###
@router.post("/upload-pdfs/")
async def upload_to_vdb(request: Request, files: List[UploadFile] = File(...)):
    directory_path = "documents"
    os.makedirs(directory_path, exist_ok=True)
    saved_file_paths = []

    for file in files:
        # contents = await file.read()
        path = os.path.join(directory_path, file.filename)
        logger.info(
            f"path: {path}",
        )
        with open(path, "wb") as f:
            shutil.copyfileobj(file.file, f)
            # f.write(contents)

        saved_file_paths.append(path)
        logger.info(f"Saved file paths: {saved_file_paths}")

    docs_after_split = load_and_split_documents(saved_file_paths)
    request.app.vdb = FAISS.from_documents(docs_after_split, embedding_model)
    request.app.vdb.save_local(vdb_path)
    # vdb_add_embeddings(docs_after_split, embedding_model)

    return JSONResponse(
        status_code=200, content={"message": "File(s) uploaded successfully."}
    )


class GenerationRequest(BaseModel):
    """Request body for the /generate API endpoint"""

    prompt: str
    temperature: float = 0.1
    max_length: int = 250
    top_k: int = 50
    top_p: float = 0.9
    num_return_sequences: int = 1
    early_stopping: bool = True


@router.post("/response-chain")
async def response_chain(request_body: GenerationRequest, request: Request) -> dict:
    llm = request.app.model.langchain_llm(
        request_body.temperature,
        request_body.max_length,
        request_body.top_k,
        request_body.top_p,
        request_body.num_return_sequences,
        request_body.early_stopping,
    )

    # request.app.vdb = FAISS.load_local(vdb_path, embedding_model)
    retriever = get_retriever(request.app.vdb)
    memory = ConversationBufferMemory(memory_key="context", return_messages=True)
    logger.info(memory.load_memory_variables({}))
    chat_history = memory.load_memory_variables({})["context"]
    logger.info(chat_history)

    if request_body.prompt:
        input = request_body.prompt
        # system_prompt = request.app.conf["system_prompt"]
        prompt_template = request.app.conf["prompt_template"]
        # update config prompt_template

        start_time = time.time()
        response = get_response_chain(
            input, prompt_template, llm, retriever, chat_history
        )

        end_time = time.time()
        latency = end_time - start_time
        print(f"Latency: {latency} seconds")

        return {"response": response}
    else:
        raise HTTPException(status_code=400, detail="Prompt cannot be empty.")
