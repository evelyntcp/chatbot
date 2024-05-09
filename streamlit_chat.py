### Import libraries ###
import os

import requests
import streamlit as st
from dotenv import load_dotenv

### Config ###
load_dotenv(".env")
backend_url = os.getenv("FASTAPI_URL")  # to inference endpoint

### Streamlit ###
st.title("Chatbot customizer")

st.markdown(
    """
This chatbot is powered by Meta Llama 3. 

Feel free to interact with the model through the inputs below and see how it responds to various prompts and settings!
"""
)

# UI elements to accept user inputs
user_prompt = st.text_input("Enter your question:")
temperature = st.number_input(
    "Temperature:",
    value=0.1,
    min_value=0.0,
    max_value=1.0,
    step=0.05,
    help="Controls the randomness of the predictions. Lower values make the model more deterministic in its responses, while higher values make it more creative.",
)
top_k = st.number_input(
    "Top K:",
    value=50,
    min_value=1,
    max_value=100,
    step=5,
    help="Limits the generated words to the K most likely next words. Set to a higher number to increase diversity.",
)
top_p = st.number_input(
    "Top P:",
    value=1.0,
    min_value=0.0,
    max_value=1.0,
    step=0.05,
    help="Applies 'nucleus sampling', filtering out the tail of the distribution to focus generation on the most likely tokens. Lower values result in more focused samples.",
)
submit_button = st.button("Generate Response")

# To add in future: chat templating option

if submit_button:
    if user_prompt:
        data = {
            "prompt": user_prompt,
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
        }

        response = requests.post(backend_url, json=data)
        if response.status_code == 200:
            chat_response = response.json()["response"]
            st.write(chat_response)
        else:
            st.write("Error in generating response")
    else:
        st.write("Please enter a prompt.")
