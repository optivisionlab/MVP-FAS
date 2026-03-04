FROM pytorch/pytorch:2.7.1-cuda11.8-cudnn9-runtime

WORKDIR /download

# Set timezone and non-interactive mode
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Ho_Chi_Minh

# Add NVIDIA repository and install CUDA libraries
RUN apt-get update && apt-get install -y \
    gnupg software-properties-common wget && \
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb && \
    dpkg -i cuda-keyring_1.0-1_all.deb && \
    apt-get update && apt-get install -y \
    ffmpeg libsm6 libxext6 build-essential libzbar-dev \
    zlib1g-dev libncurses5-dev libgdbm-dev libnss3-dev \
    libssl-dev libreadline-dev libffi-dev libsqlite3-dev wget \
    libbz2-dev vim tzdata cuda-libraries-11-1 cuda-nvrtc-11-1 \
    cuda-nvrtc-dev-11-1 && \
    ln -fs /usr/share/zoneinfo/$TZ /etc/localtime && \
    dpkg-reconfigure --frontend noninteractive tzdata \
    && apt install -y cmake

RUN apt-get install vim -y

# Reset DEBIAN_FRONTEND
ENV DEBIAN_FRONTEND=

# Install Python 3.8
# RUN wget https://www.python.org/ftp/python/3.12.12/Python-3.12.12.tgz \
#     && tar -xf Python-3.12.12.tgz

# WORKDIR /download/Python-3.12.12
# RUN ./configure --enable-optimizations && make -j 8 && make altinstall

WORKDIR /app

# Upgrade pip and install PyTorch and Detectron2
RUN python3.11 -m pip install --upgrade pip
RUN pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118
# RUN pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu111/torch1.8/index.html

WORKDIR /app/sources

# Set environment variables
ENV PYTHONPATH=/app/:$PYTHONPATH
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH

COPY requirements.txt requirements.txt

RUN pip install -r requirements.txt
# CMD ["python3.8", "ocr/main.py"]
