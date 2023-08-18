# Use the Alpine Linux base image
FROM alpine:latest

# Install necessary packages and clean up
RUN apk update \
    && apk add --no-cache \
        git \
        sudo \
        build-base \
        dpkg-dev \
        wget \
        ca-certificates \
        tzdata \
    && rm -rf /var/cache/apk/*

# Set the desired locale environment variable
ENV LANG en_US.UTF-8

# Clone the repository and download the model
RUN git clone https://github.com/ggerganov/llama.cpp.git /llama.cpp
WORKDIR /llama.cpp
RUN mkdir -p models/7B

# Build the project
RUN make

# Download the ggml formatted model checkpoint
RUN wget https://huggingface.co/TheBloke/LLaMa-7B-GGML/resolve/main/llama-7b.ggmlv3.q4_0.bin -O models/7B/ggml-model.bin

# Set the entry point for the container
CMD ["./server", "--host", "0.0.0.0"]
