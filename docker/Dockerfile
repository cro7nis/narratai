FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime
LABEL maintainer="cro7 <cro7nis@gmail.com>"

ENV APP=narratai
ENV GRADIO_SERVER_NAME=0.0.0.0
ENV GRADIO_SERVER_PORT=7860

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6 libjpeg-dev libpng-dev -y

RUN mkdir -p /opt/APP

WORKDIR /opt/$APP
RUN pip install poetry
COPY pyproject.toml pyproject.toml
#COPY  poetry.lock  poetry.lock
RUN  poetry config virtualenvs.create false \
    && poetry install --no-root --no-interaction \
    && rm -rf /root/.cache/pypoetry

#COPY ckpt/ ckpt/
COPY configs/ configs/
COPY assets/ assets/
COPY checkpoints/ checkpoints/
COPY src/ .

ENV LD_LIBRARY_PATH=/opt/conda/lib/python3.11/site-packages/nvidia/cublas/lib:/opt/conda/lib/python3.11/site-packages/nvidia/cudnn/lib

ENTRYPOINT ["python", "app.py"]