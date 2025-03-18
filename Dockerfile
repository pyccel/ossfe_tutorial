FROM python:3.11-slim

RUN apt update && apt install -y --no-install-recommends \
    gcc \
    gfortran \
    libmpich-dev \
    git \
    build-essential \
 && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip

RUN pip install pyccel

RUN mkdir /workspace
VOLUME /workspace
WORKDIR /workspace

CMD ["/bin/bash"]

