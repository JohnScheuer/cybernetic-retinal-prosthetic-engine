FROM nvidia/cuda:12.8.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    g++ \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install numpy pandas matplotlib

WORKDIR /app
COPY . /app

RUN mkdir -p build && cd build && cmake .. && make -j$(nproc)

CMD ["./build/retinal_prosthetic_engine"]
