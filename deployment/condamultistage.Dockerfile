# Stage 1 from nvidia CUDA + cuDNN base image
# FROM nvidia/cuda:12.3.2-cudnn9-runtime-ubuntu22.04 as base
FROM nvcr.io/nvidia/pytorch:23.11-py3 as base
# already has python3.10, CUDA, cuDNN, and torch

LABEL maintainer="Evelyn"

ARG DEBIAN_FRONTEND=noninteractive
ARG USER_ID=1000
ARG GROUP_ID=3000

RUN apt-get update && \
    apt-get install -y \
    git wget unzip bzip2 build-essential ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Stage 2 from miniconda image
FROM continuumio/miniconda3 as conda
COPY conda-env.yaml conda-env.yaml
RUN conda env create -f conda-env.yaml

FROM base as final

COPY --from=conda /opt/conda /opt/conda

RUN addgroup --gid $GROUP_ID appgroup && \
    adduser --disabled-password --gecos '' --uid $USER_ID --gid $GROUP_ID appuser

COPY --chown=appuser:appgroup conf/base/model.yaml /home/appuser/app/conf/base/model.yaml
COPY --chown=appuser:appgroup conf/base/logging.yaml /home/appuser/app/conf/base/logging.yaml
COPY --chown=appuser:appgroup src/model.py /home/appuser/app/src/model.py
COPY --chown=appuser:appgroup src/utils.py /home/appuser/app/src/utils.py
COPY --chown=appuser:appgroup main.py /home/appuser/app/main.py
COPY --chown=appuser:appgroup streamlit_chat.py /home/appuser/app/streamlit_chat.py
# COPY --chown=appuser:appgroup Meta-Llama-3-8B-Instruct /home/appuser/app/Meta-Llama-3-8B-Instruct

RUN pip install streamlit

RUN chgrp -R $GROUP_ID /home/appuser/app && \
    chmod -R g+rwX /home/appuser/app

USER $USER_ID:$GROUP_ID

# ENV PATH="/home/appuser/.local/bin:${PATH}"
ENV PATH="/opt/conda/envs/chat2/bin:${PATH}"
# ENV PYTHONPATH="${PYTHONPATH}:home/appuser/app"

WORKDIR /home/appuser/app

EXPOSE 8000
EXPOSE 8501