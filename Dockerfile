FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-devel


ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update -qq \
 && apt-get install --no-install-recommends -y \
    # install essentials
    build-essential \
    g++ \
    git \
    openssh-client \
    tmux \
    vim \
    libgl1-mesa-glx\
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*