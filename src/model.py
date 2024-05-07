import logging
import os
import time
from typing import List, Union

import fastapi
import torch
import yaml
from fastapi import Depends, FastAPI, HTTPException
from transformers import AutoModelForCausalLM, AutoTokenizer

### Initialization ###

logger = logging.getLogger(__name__)

router = fastapi.APIRouter()

### Config variables ###
with open("conf/base/model.yaml") as f:
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
        if not os.path.isdir(model_path):
            logger.debug("model_path does not exist. Downloading of model will begin.")
            self.tokenizer, self.model = download_hf_model()
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForCausalLM.from_pretrained(model_path)
        except Exception as e:
            logger.error(
                f"An unexpected error occurred while loading the model from '{model_path}'. Error: {e}"
            )
            raise ValueError("An unexpected error occurred.") from e

    def generate_response(
        self, prompt: str, config: dict, num_return_sequences: int = 1
    ) -> Union[str, List[str]]:
        """
        Generates a response to a given prompt, formatted with
        a predefined system prompt extracted from the configuration, utilizing the specified generation parameters.

        Args:
            prompt (str): The user's input to which the model will generate a response.
            config (dict): A dictionary containing the generation parameters, including
                'max_length', 'min_length', 'temperature', 'top_k', 'top_p',
                'no_repeat_ngram_size', and 'early_stopping'.
            num_return_sequences (int): The number of response sequences to generate
                from the given prompt. Default is 1.

        Returns:
            Union[str, List[str]]: A single response string if 'num_return_sequences' is 1,
                otherwise a list of response strings.

        The generated response(s) leverage a 'system' role declaration to simulate a
        conversational context with the AI, aiming for interactions where the AI
        embodies a character as per the configured system prompt.

        Execution time is reported to give an indication of the response latency.
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
            max_length=config["max_length"],
            min_length=config["min_length"],
            temperature=config["temperature"],
            top_k=config["top_k"],
            top_p=config["top_p"],
            num_return_sequences=num_return_sequences,
            early_stopping=config["early_stopping"],
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

    with open("conf/secrets.yaml") as secrets_file:
        secrets = yaml.safe_load(secrets_file)
        hf_token = secrets.get("HF_TOKEN")
    with open("conf/base/model.yaml") as model_file:
        args = yaml.safe_load(model_file)
        model_id = args["model_id"]
    if model_id is None:
        raise ValueError("No `model_id` provided.")

    save_model_path = f"models/download/{model_id}"

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

        tokenizer.save_pretrained(save_model_path)
        model.save_pretrained(save_model_path)
    return tokenizer, model
