import logging
import os
import time
from typing import List, Union

import fastapi
import torch
import yaml
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from transformers import AutoModelForCausalLM, AutoTokenizer

### Initialization ###

logger = logging.getLogger(__name__)

router = fastapi.APIRouter()

load_dotenv(".env")

## Config variables ###
with open("conf/base/model.yaml") as f:
    global conf  # in colab i didnt have this
    conf = yaml.safe_load(f)


### Model Class ###
class Model:
    """A class to handle model initialization and response generation.

    Attributes:
        tokenizer (AutoTokenizer): The tokenizer for the model.
        model (AutoModelForCausalLM): The language model.

    Methods:
        generate_response(prompt: str, max_length: int, num_return_sequences: int): Generates a response to a given prompt.
    """

    def __init__(self, model_path: str) -> None:
        try:
            if not os.path.isdir(model_path):
                logger.debug(
                    "model_path does not exist. Downloading of model will begin."
                )
                self.tokenizer, self.model = download_hf_model()
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
        temperature: float = 0.1,
        top_k: int = 50,
        top_p: float = 0.9,
        num_return_sequences: int = 1,
    ) -> Union[str, List[str]]:
        """
        Generates a response to a given prompt using the model.

        Args:
            prompt (str): The user's input to which the model will generate a response.
            temperature (float): Controls the randomness in the response generation. Lower values
                make the responses more deterministic. Default is 0.1.
            top_k (int): Limits the number of highest probability vocabulary tokens considered for
                generating responses. Default is 50.
            top_p (float): Nucleus sampling. Keeps the top tokens with cumulative probability
                greater than this value. Default is 0.9.
            num_return_sequences (int): The number of response sequences to generate
                from the given prompt. Default is 1.

        Returns:
            Union[str, List[str]]: A single response string if 'num_return_sequences' is 1,
                otherwise a list of response strings. Each response is generated based on the
                given prompt and generation parameters.
        """

        start_time = time.time()

        # system_prompt = conf["system_prompt"]
        # full_prompt = f"[system]: {system_prompt}\n[user]: {prompt}"

        # input_ids = self.tokenizer.encode(full_prompt, return_tensors="pt")
        # prompt = self.tokenizer.decode(input_ids[0].tolist())

        # input_ids = self.tokenizer.apply_chat_template(
        #     full_prompt, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        # )
        # print(self.tokenizer.decode(input_ids[0]))

        input_ids = self.tokenizer.encode(prompt, return_tensors="pt")
        output = self.model.generate(
            input_ids,
            max_length=conf["max_length"],
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            num_return_sequences=num_return_sequences,
            early_stopping=conf["early_stopping"],
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
def download_hf_model() -> tuple:
    """Downloads and saves the model and tokenizer based on the configuration files.

    The configuration should specify `HF_TOKEN` and `model_id`.

    Returns:
        tuple: The tokenizer and model that were downloaded.

    Raises:
        ValueError: If no `model_id` is specified in the configuration.
    """
    tokenizer = None
    model = None

    # with open("conf/secrets.yaml") as secrets_file:
    #     secrets = yaml.safe_load(secrets_file)
    #     hf_token = secrets.get("HF_TOKEN")
    hf_token = os.getenv("HF_TOKEN")
    with open("conf/base/model.yaml") as model_file:
        conf = yaml.safe_load(model_file)
        model_id = conf["model_id"]
    if model_id is None:
        raise ValueError("No `model_id` provided.")

    save_model_path = conf["save_model_path"]

    if not os.path.exists(save_model_path):
        os.makedirs(save_model_path, exist_ok=True)

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            load_in_4bit=True,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            cache_dir="models",
            token=hf_token,
            # bnb_4bit_compute_dtype=torch.float16,
            # bnb_4bit_quant_type="nf4",
            # bnb_4bit_use_double_quant=True
        )
        tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)

        model.save_pretrained(save_model_path)
        tokenizer.save_pretrained(save_model_path)
    return tokenizer, model
