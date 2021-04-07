FROM pytorch/pytorch:1.6.0-cuda10.1-cudnn7-runtime

WORKDIR /app

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

ENV APP_ROOT /app
ENV WORK_DIR /app/workspace

ENV DEBIAN_FRONTEND noninteractive

RUN mkdir -p $APP_ROOT
WORKDIR $APP_ROOT

RUN ln -sf /usr/share/zoneinfo/Asia/Tokyo /etc/localtime

RUN apt-get update && apt-get install -y \
    build-essential \
    sudo \
    git \
    wget \
    curl \
    cmake \
    file \
    unzip \
    gcc \
    g++ \
    xz-utils \
    mecab \
    libmecab-dev \
    mecab-ipadic-utf8 \
    python3-pip \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
RUN ln -s /etc/mecabrc /usr/local/etc/mecabrc
RUN pip3 install -U pip

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY data.py /app/data.py
COPY tokenizer.py /app/tokenizer.py
COPY model.py /app/model.py
COPY datamodule.py /app/datamodule.py
COPY trf_main.py /app/trf_main.py

CMD ["python3", "/app/trf_main.py"]