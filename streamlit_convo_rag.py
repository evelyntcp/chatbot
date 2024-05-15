### Import libraries ###
import logging
import os
import time

import requests
import streamlit as st
from dotenv import load_dotenv

### Initialization ###
load_dotenv()
retrieval_url = os.getenv("RETRIEVAL_URL")
upload_pdf_url = os.getenv("UPLOAD_URL")
normal_response_url = os.getenv("RESPONSE_URL")

logger = logging.getLogger(__name__)


### Functions ###
def send_prompt_and_get_response(prompt, temp=0.1, k=50, p=1.0, uploaded_files=False):
    with st.chat_message("assistant"):
        with st.spinner("Generating......"):
            data = {
                "prompt": prompt,
                "temperature": temp,
                "top_k": k,
                "top_p": p,
            }
            if uploaded_files:
                logger.info("Retrieving")
                response = requests.post(retrieval_url, json=data)
            else:
                logger.info("Normal chat")
                response = requests.post(normal_response_url, json=data)

            if response.status_code == 200:
                return response.json()["response"]
            else:
                return "Sorry, there was an error generating a response."


def clear_history():
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "You can upload a PDF and ask me questions about it, or you can just chat with me!",
        }
    ]


### Streamlit ###
st.set_page_config(page_title="Chat + Documents!")

uploaded_file = st.sidebar.file_uploader(
    "Choose PDF file",
    accept_multiple_files=True,
    type=["pdf"],
)

if uploaded_file:
    st.session_state.uploaded_file = True
    # print(uploaded_file)
    # files = {"file": uploaded_file} # single file
    files = [("files", file) for file in uploaded_file]
    response = requests.post(
        upload_pdf_url,
        files=files,
    )
    if response.status_code == 200:
        st.sidebar.write("Upload successful.")
    else:
        st.sidebar.write("Error in generating response")
else:
    st.session_state.uploaded_file = False

st.markdown(
    """
This chatbot is powered by Meta Llama 3.
"""
)

prompt = st.chat_input("Ask me anything about your document.")

st.sidebar.button("Clear Chat History", on_click=clear_history)
temperature = st.sidebar.number_input(
    "Temperature:",
    value=0.1,
    min_value=0.0,
    max_value=1.0,
    step=0.05,
    help="Controls the randomness of the predictions. Lower values make the model more deterministic in its responses, while higher values make it more creative.",
)
top_k = st.sidebar.number_input(
    "Top K:",
    value=50,
    min_value=1,
    max_value=100,
    step=5,
    help="Limits the generated words to the K most likely next words. Set to a higher number to increase diversity.",
)
top_p = st.sidebar.number_input(
    "Top P:",
    value=1.0,
    min_value=0.0,
    max_value=1.0,
    step=0.05,
    help="Applies 'nucleus sampling', filtering out the tail of the distribution to focus generation on the most likely tokens. Lower values result in more focused samples.",
)


if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "You can upload a PDF and ask me questions about it, or you can just chat with me!",
        }
    ]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)
    # if "uploaded_files" not in st.session_state:
    #     st.session_state.uploaded_files = uploaded_file


if st.session_state.messages[-1]["role"] != "assistant":
    placeholder = st.empty()
    if uploaded_file:
        assistant_response = send_prompt_and_get_response(
            prompt,
            temperature,
            top_k,
            top_p,
            uploaded_files=True,
        )
    else:
        assistant_response = send_prompt_and_get_response(
            prompt,
            temperature,
            top_k,
            top_p,
            uploaded_files=False,
        )
    placeholder.markdown(assistant_response)
    st.session_state.messages.append(
        {"role": "assistant", "content": assistant_response}
    )

    # uploaded_files = st.session_state.get("uploaded_files")
    # assistant_response = send_prompt_and_get_response(
    #     prompt, uploaded_files=uploaded_files
    # )
    # st.session_state.messages.append(
    #     {"role": "assistant", "content": assistant_response}
    # )
    # st.session_state.prompt = ""

# st.chat_message("")
st.session_state.messages = st.session_state.messages[-5:]
