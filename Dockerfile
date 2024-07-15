FROM nvcr.io/nvidia/pytorch:22.12-py3

ENV DEBIAN_FRONTEND=noninteractive
ARG USERNAME=user
ARG WORKDIR=/workspace/ZeroAI-Refactor

RUN apt-get update && apt-get install -y \
        automake autoconf libpng-dev nano python3-pip \
        curl zip unzip libtool swig zlib1g-dev pkg-config \
        python3-mock libpython3-dev libpython3-all-dev \
        g++ gcc cmake make pciutils cpio gosu wget \
        libgtk-3-dev libxtst-dev sudo apt-transport-https \
        build-essential gnupg git xz-utils vim \
        libva-drm2 libva-x11-2 vainfo libva-wayland2 libva-glx2 \
        libva-dev libdrm-dev xorg xorg-dev protobuf-compiler \
        openbox libx11-dev libgl1-mesa-glx libgl1-mesa-dev \
        libtbb2 libtbb-dev libopenblas-dev libopenmpi-dev \
    && sed -i 's/# set linenumbers/set linenumbers/g' /etc/nanorc \
    && apt clean \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

RUN git clone https://github.com/congtoudada/ZeroAI-Refactor.git \
    && cd ZeroAI-Refactor \
    && git checkout c3a32e85218e830b702a1012c0e0029cb1aa811f \
    && pip3 install pip --upgrade \
    && pip3 install -r requirements.txt \

RUN python3 installer.py \
    && ldconfig \
    && pip cache purge

RUN git clone https://github.com/NVIDIA-AI-IOT/torch2trt.git \
    && cd torch2trt \
    && git checkout bf84e1aa24522bfe047a7eedf8430e17ef801936 \
    && python3 setup.py install

RUN echo "root:root" | chpasswd \
    && adduser --disabled-password --gecos "" "${USERNAME}" \
    && echo "${USERNAME}:${USERNAME}" | chpasswd \
    && echo "%${USERNAME}    ALL=(ALL)   NOPASSWD:    ALL" >> /etc/sudoers.d/${USERNAME} \
    && chmod 0440 /etc/sudoers.d/${USERNAME}
USER ${USERNAME}
RUN sudo chown -R ${USERNAME}:${USERNAME} ${WORKDIR}
WORKDIR ${WORKDIR}