# FROM nvidia/cuda:12.3.0-base-ubuntu22.04
FROM nvidia/cuda:12.3.2-cudnn9-runtime-ubuntu22.04

LABEL maintainer="Evelyn"

ARG DEBIAN_FRONTEND=noninteractive
ARG USER_ID=1000
ARG GROUP_ID=3000

RUN apt-get update && \
    apt-get install -y \
    git \
    python3 \
    python3-pip && \
    python3 -m pip install --no-cache-dir --upgrade pip && \
    rm -rf /var/lib/apt/lists/*

RUN addgroup --gid $GROUP_ID appgroup && \
    adduser --disabled-password --gecos '' --uid $USER_ID --gid $GROUP_ID appuser

COPY --chown=appuser:appgroup requirements.txt /home/appuser/app/requirements.txt
# RUN apt-get install -y \
#     python3-venv \
#     && python3 -m venv /opt/venv  \
#     && export PATH=/opt/venv/bin:$PATH \
#     && echo "source /opt/venv/bin/activate" >> ~/.bashrc \
#     && source /opt/venv/bin/activate \
RUN python3 -m pip install --no-cache-dir -r /home/appuser/app/requirements.txt

COPY --chown=appuser:appgroup conf/base/model.yaml /home/appuser/app/conf/base/model.yaml
COPY --chown=appuser:appgroup conf/base/logging.yaml /home/appuser/app/conf/base/logging.yaml
COPY --chown=appuser:appgroup src/model.py /home/appuser/app/src/model.py
COPY --chown=appuser:appgroup src/utils.py /home/appuser/app/src/utils.py
COPY --chown=appuser:appgroup main.py /home/appuser/app/main.py
# COPY --chown=appuser:appgroup streamlit_chat.py /home/appuser/app/streamlit_chat.py
COPY --chown=appuser:appgroup streamlit_convo_rag.py /home/appuser/app/streamlit_convo_rag.py
COPY --chown=appuser:appgroup src/rag.py /home/appuser/app/src/rag.py

RUN chgrp -R $GROUP_ID /home/appuser/app && \
    chmod -R g+rwX /home/appuser/app

USER $USER_ID:$GROUP_ID

ENV PATH="/home/appuser/.local/bin:${PATH}"
ENV PYTHONPATH="${PYTHONPATH}:home/appuser/app"

WORKDIR /home/appuser/app

EXPOSE 8000
EXPOSE 8501

ENTRYPOINT ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port 8000 & streamlit run streamlit_convo_rag.py --server.port 8501"]
# CMD, can put the port args in CMD too