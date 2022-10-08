# CUDA 11.0, CUDNN 8.0, Tensorflow, PyTorch, zsh, pyenv, vnc
# See -> https://hub.docker.com/r/naruya/dl_remote

# [1] https://github.com/robbyrussell/oh-my-zsh
# [2] https://github.com/pyenv/pyenv/wiki/common-build-problems
# [3] https://github.com/tensorflow/tensorflow/blob/master/tensorflow/tools/dockerfiles/dockerfiles/devel-gpu-jupyter.Dockerfile


# CUDA (from [3]) ################################
ARG UBUNTU_VERSION=20.04

ARG ARCH=
ARG CUDA=11.3
FROM nvidia/cuda${ARCH:+-$ARCH}:${CUDA}.1-base-ubuntu${UBUNTU_VERSION} as base

ENV TZ=Asia/Tokyo
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

################################
# ----------------------------------------------------------------------------------------------------------------------
#
#   User Setup
#
# ----------------------------------------------------------------------------------------------------------------------
# Define versions.
ARG SUDO_VERSION=*

# Install sudo
RUN apt-get update && apt-get install -y --no-install-recommends \
    sudo=${SUDO_VERSION} \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Root user setting
ENV ROOT_USER=root
ARG ROOT_USER_PASSWORD=root

# root password is root
RUN echo ${ROOT_USER}:${ROOT_USER_PASSWORD} | chpasswd
WORKDIR /root

# zsh (from [1]) for root----------------
RUN apt-get update -q \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    wget \
    git \
    zsh \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

SHELL ["/bin/zsh", "-c"]
RUN wget http://github.com/robbyrussell/oh-my-zsh/raw/master/tools/install.sh -O - | zsh


# Add default user
ENV DEFAULT_USER=docker
ENV DEFAULT_USER_HOME=/home/${DEFAULT_USER}
ARG DEFAULT_USER_PASSWORD=password
ARG DEFAULT_USER_TEMP_ID=9999
ARG DEFAULT_USER_GROUP_NAME=docker
ARG DEFAULT_USER_GROUP_TEMP_ID=65534

RUN groupadd -g ${DEFAULT_USER_GROUP_TEMP_ID} ${DEFAULT_USER_GROUP_NAME} && \
    useradd -m -s /bin/bash -u ${DEFAULT_USER_TEMP_ID} -g ${DEFAULT_USER_GROUP_TEMP_ID} -G sudo ${DEFAULT_USER} && \
    echo ${DEFAULT_USER}:${DEFAULT_USER_PASSWORD} | chpasswd && \
    echo "${DEFAULT_USER}   ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

USER ${DEFAULT_USER_TEMP_ID}
WORKDIR ${DEFAULT_USER_HOME}

# zsh (from [1]) ----------------
RUN sudo apt-get update -q \
    && DEBIAN_FRONTEND=noninteractive sudo apt-get install -y \
    wget \
    git \
    zsh \
    && sudo apt-get clean \
    && sudo rm -rf /var/lib/apt/lists/*

SHELL ["/bin/zsh", "-c"]
RUN wget http://github.com/robbyrussell/oh-my-zsh/raw/master/tools/install.sh -O - | zsh

# pyenv (from [2]) ----------------
ENV DEBIAN_FRONTEND=noninteractive
RUN sudo apt-get update && sudo apt-get install -y \
    build-essential libssl-dev zlib1g-dev libbz2-dev libreadline-dev \
    libsqlite3-dev wget curl llvm libncurses5-dev libncursesw5-dev \
    xz-utils tk-dev libffi-dev liblzma-dev python-openssl git
RUN curl https://pyenv.run | zsh && \
    echo 'export PATH="${DEFAULT_USER_HOME}/.pyenv/bin:$PATH"' >> ${DEFAULT_USER_HOME}/.zshrc && \
    echo 'eval "$(pyenv init -)"' >> ${DEFAULT_USER_HOME}/.zshrc && \
    echo 'eval "$(pyenv virtualenv-init -)"' >> ${DEFAULT_USER_HOME}/.zshrc
RUN source ${DEFAULT_USER_HOME}/.zshrc && \
    pyenv install 3.8.0 && \
    pyenv global 3.8.0 && \
    pip install -U pip

# X window ----------------
RUN sudo apt-get update && sudo apt-get install -y \
    xvfb x11vnc python-opengl icewm \
    && sudo apt-get clean \
    && sudo rm -rf /var/lib/apt/lists/*

# utils ----------------
RUN sudo apt-get update -q \
    && DEBIAN_FRONTEND=noninteractive sudo apt-get install -y \
    curl \
    git \
    unzip \
    vim \
    wget \
    ffmpeg \
    tmux \
    && sudo apt-get clean \
    && sudo rm -rf /var/lib/apt/lists/*

COPY .tmux.conf ${DEFAULT_USER_HOME}/

# DL ----------------
RUN source ${DEFAULT_USER_HOME}/.zshrc && \
    pip install \
    setuptools \
    jupyterlab \
    widgetsnbextension \
    && \
    echo 'alias jl="DISPLAY=:0 jupyter lab --ip 0.0.0.0 --port 8888 --allow-root &"' >> ${DEFAULT_USER_HOME}/.zshrc && \
    echo 'alias tb="tensorboard --logdir runs --bind_all &"' >> ${DEFAULT_USER_HOME}/.zshrc

# Pytorch ----------------
RUN source ${DEFAULT_USER_HOME}/.zshrc && \
    pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113

# Experiment Manager
RUN source ${DEFAULT_USER_HOME}/.zshrc && \
    pip install \
    mlflow \
    hydra-core \
    wandb
    
# Utils
RUN source ${DEFAULT_USER_HOME}/.zshrc && \
    pip install \
    joblib \
    scikit-image

RUN sudo apt-get clean && \
    sudo rm -rf /var/lib/apt/lists/*

COPY start.sh ${DEFAULT_USER_HOME}
COPY .tmux.conf ${DEFAULT_USER_HOME}


RUN echo 'Xvfb :0 -screen 0 1400x900x24 & ' >> ${DEFAULT_USER_HOME}/Xvfb-run.sh && \
    echo 'x11vnc -display :0 -passwd pass -forever &' >> ${DEFAULT_USER_HOME}/run-Xvfb.sh && \
    chmod +x ${DEFAULT_USER_HOME}/run-Xvfb.sh

RUN echo 'DISPLAY=:0 jupyter notebook --allow-root --ip=0.0.0.0 --port 8888 --notebook-dir=${DEFAULT_USER_HOME} --NotebookApp.password="sha1:71247b1fba50:6334281a44d2134e85492be9ad7426a3cf9caf90" &' >> ${DEFAULT_USER_HOME}/run-jupyter.sh && \
    chmod +x ${DEFAULT_USER_HOME}/run-jupyter.sh

RUN sudo cp ${DEFAULT_USER_HOME}/.zshrc /root/.zshrc

# auto start tmux and zsh
ENTRYPOINT tmux new \; \
            send-keys 'Xvfb :0 -screen 0 1400x900x24 & ' Enter \; \
	    send-keys 'x11vnc -display :0 -passwd 0123 -forever &' Enter \; \
            split-window -v  \; \
            send-keys "zsh" Enter \; \
            send-keys "jupyter nbextension enable --py widgetsnbextension --sys-prefix" Enter \; \
            send-keys "bash ${DEFAULT_USER_HOME}/run-jupyter.sh" Enter \; \
	   new-window \; \
            send-keys "zsh" Enter \; \
    	    send-keys clear C-m \;