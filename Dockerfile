# syntax=docker/dockerfile:1

FROM python:3.9-slim-buster

COPY requirements_pip.txt /opt/program/requirements.txt


# Install base utilities
RUN apt-get update \
    && apt-get install -y build-essential \
    && apt-get install -y wget \
    && apt-get clean \
    && apt install gcc \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get -y update && apt-get install -y --no-install-recommends \
         wget \
         nginx \
         ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# install requirements
RUN python3 -m pip install --upgrade pip setuptools wheel
RUN python3 -m pip install -r /opt/program/requirements.txt

# set the paths
ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV PATH="/opt/program:${PATH}"
ENV MODEL_PATH="/opt/ml/model"

# Set up the program in the image
COPY src /opt/program
RUN chmod 755 /opt/program

WORKDIR /opt/program

# make serve and train executable
RUN chmod 755 serve
RUN chmod 755 train
