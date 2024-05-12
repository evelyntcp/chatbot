import logging
import os
import time
from typing import List, Union

import fastapi
import torch
import yaml

# from dotenv import load_dotenv
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer

### Initialization ###

logger = logging.getLogger(__name__)

router = APIRouter()

model = None

# load_dotenv(".env")

torch_device = "cuda" if torch.cuda.is_available() else "cpu"


### Model Class ###
class Model:
    """A class to handle model initialization and response generation.

    Attributes:
        tokenizer (AutoTokenizer): The tokenizer for the model.
        model (AutoModelForCausalLM): The language model.

    Methods:
        generate_response(prompt: str, max_length: int, num_return_sequences: int): Generates a response to a given prompt.
    """

    def __init__(self, model_id: str, model_path: str) -> None:
        try:
            if not os.path.isdir(model_path):
                logger.debug(
                    "Model path does not exist. Downloading of model will begin."
                )
                self.tokenizer, self.model = download_hf_model(model_id, model_path)
                logger.info("Model and tokenizer successfully downloaded.")
            else:
                self.tokenizer = AutoTokenizer.from_pretrained(model_path)
                self.model = AutoModelForCausalLM.from_pretrained(model_path)
                logger.info("Model and tokenizer loaded.")

        except Exception as e:
            logger.error(
                f"An unexpected error occurred while loading the model from '{model_path}'. Error: {e}"
            )
            raise ValueError("An unexpected error occurred.") from e

    def generate_response(
        self,
        prompt: str,
        max_length: int = 500,
        temperature: float = 0.1,
        top_k: int = 50,
        top_p: float = 0.9,
        num_return_sequences: int = 1,
        early_stopping: bool = True,
    ) -> Union[str, List[str]]:
        """
        Generates a response to a given prompt using the model.

        Args:
            prompt (str): The user's input to which the model will generate a response.
            max_length (int): The maximum length of the generated text. Default is 250.
            temperature (float): Controls the randomness in the response generation. Lower values
                make the responses more deterministic. Default is 0.1.
            top_k (int): Limits the number of highest probability vocabulary tokens considered for
                generating responses. Default is 50.
            top_p (float): Nucleus sampling. Keeps the top tokens with cumulative probability
                greater than this value. Default is 0.9.
            num_return_sequences (int): The number of response sequences to generate
                from the given prompt. Default is 1.
            early_stopping (bool): Whether or not to stop generating once `early_stopping`
                criteria are met. If set to False, the generator will keep generating until it reaches
                `max_length`. Default is True.

        Returns:
            Union[str, List[str]]: A single response string if 'num_return_sequences' is 1,
                otherwise a list of response strings. Each response is generated based on the
                given prompt and generation parameters.
        """

        start_time = time.time()

        # input_ids = self.tokenizer.apply_chat_template(
        #     full_prompt, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        # )

        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(torch_device)
        output = self.model.generate(
            input_ids,
            max_length=max_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            num_return_sequences=num_return_sequences,
            early_stopping=early_stopping,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        responses = [
            self.tokenizer.decode(output_seq, skip_special_tokens=True)
            for output_seq in output
        ]
        end_time = time.time()
        latency = end_time - start_time
        print(f"Latency: {latency} seconds")
        print("Generated text:", responses)

        return responses[0] if num_return_sequences == 1 else responses


### Functions ###
def download_hf_model(model_id, save_model_path) -> tuple:
    """Downloads and saves the model and tokenizer based on the configuration files.

    The configuration should specify `HF_TOKEN` and `model_id`.

    Returns:
        tuple: The tokenizer and model that were downloaded.

    Raises:
        ValueError: If no `model_id` is specified in the configuration.
    """
    tokenizer = None
    model = None
    hf_token = os.getenv("HF_TOKEN")

    if model_id is None:
        raise ValueError("No `model_id` provided.")

    if not os.path.exists(save_model_path):
        os.makedirs(save_model_path, exist_ok=True)

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            load_in_4bit=True,
            device_map="auto",
            cache_dir="models",
            token=hf_token,
            bnb_4bit_compute_dtype=torch.float16,
            # bnb_4bit_quant_type="nf4",
            # bnb_4bit_use_double_quant=True
        )
        tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)

        model.save_pretrained(save_model_path)
        tokenizer.save_pretrained(save_model_path)
    return tokenizer, model


### APIs ###
class GenerationRequest(BaseModel):
    """Request body for the /generate API endpoint"""

    prompt: str
    max_length: int = 250
    temperature: float = 0.1
    max_length: int = 250
    top_k: int = 50
    top_p: float = 0.9
    num_return_sequences: int = 1
    early_stopping: bool = True


@router.post("/response/")
async def response(request_body: GenerationRequest, request: Request) -> dict:
    """
    Process the generation request and return the response.

    Args:
    request (GenerationRequest): The request object containing the prompt,
    max length, and number of return sequences.

    Returns:
    dict: A dictionary containing the generated response.

    Raises:
    HTTPException: Raised if the prompt is empty.
    """
    if request_body.prompt:
        system_prompt = request.app.conf["system_prompt"]
        prompt_template = request.app.conf["prompt_template"]
        user_input = request_body.prompt

        try:
            full_prompt = prompt_template.format(system=system_prompt, user=user_input)
        except:
            full_prompt = user_input

        response = request.app.model.generate_response(
            full_prompt,
            request_body.max_length,
            request_body.temperature,
            request_body.top_k,
            request_body.top_p,
            request_body.num_return_sequences,
            request_body.early_stopping,
        )
        return {"response": response}
    else:
        raise HTTPException(status_code=400, detail="Prompt cannot be empty.")
