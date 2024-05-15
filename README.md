# Chatbot with Llama 3
Prototype of a chatbot using Huggingface, FastAPI and Streamlit

## Features
- Chat with PDF via retrieval augmented generation. 
- Supports normal chat with LLM.
- Can customize simple settings of the LLM such as temperature, top_k and top_p.
- Able to load local LLM and run on GPU.

## Design Flow
Streamlit interface takes in pdf file then posts it to /upload-pdfs API endpoint. 
This endpoint contains a pipeline where it loads the PDF, splits the document by chunking, then uses an embedding model to convert it to embeddings before adding to the vector database.

After the vector database is populated, users can ask questions at the streamlit interface. The streamlit app will post the user's input to another /response-chain endpoint where the pipeline will retrieve similar information and then generate a response.

In the event where there is no PDF uploaded, a normal chat can still take place between the user and chatbot. 

## Set up environment

Using virtual environment locally:
```
python -m venv chat
source chat/bin/activate
pip install -r requirements.txt
```

<!-- Using conda (for multistage docker build)

```
conda env create -f conda-env.yaml
```

If already environment is already created,

```
conda activate chat2
``` -->

## Deployment

To run docker with GPU, ensure that Nvidia GPU drivers, CUDA, Container toolkit are installed on machine. 

If installed correctly, running the below docker run command will pull an nvidia cuda image and run 'nvidia-smi'. The GPU utilization will be shown. 

```
docker run --rm --gpus all nvidia/cuda:12.3.30-base-ubuntu20.04 nvidia-smi
```

<!-- ### Docker Base Image
```
docker build -t base:latest -f deployment/base.Dockerfile .
```
Upon successful build, run below command to see GPU utilization details.
```
docker run --rm --gpus all base:latest nvidia-smi
``` -->

### Docker App Image
```
docker-compose -f deployment/rag-docker-compose.yml up
```

<!-- 
or 

docker build -t app:latest -f deployment/condamultistage.Dockerfile .
docker run -p 8000:8000 -p 8501:8501 --env-file .env app:latest -->

<!-- ### Kubernetes
Kubernetes using minikube on local PC:

```
kubectl apply -f deployment/deployment.yaml
``` -->
