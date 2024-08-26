ARG version=standard
ARG build=0

##########################
####### Base Image #######
##########################

FROM nvidia/cuda:12.5.1-devel-ubuntu22.04 AS base

ARG USERNAME=dev
ARG CODE_LOCATION=/src/DSI-Bench

# install basic packages
RUN apt update -y &&\
    apt install -y \
        vim \
        htop \
        git \
        tmux \
        wget \
        binutils \
        build-essential \
        g++ && \
    apt clean && \
    rm -rf /var/lib/apt/lists/*

# set password of root to root
RUN echo 'root:root' | chpasswd

# create user dev
RUN groupadd ${USERNAME}
RUN useradd -m -g ${USERNAME} -p admin -s /bin/bash ${USERNAME}

# create directory to install mamba
RUN mkdir /mamba
RUN chown ${USERNAME}:${USERNAME} /mamba

USER ${USERNAME}

# install mamba
RUN wget -P /tmp/ https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh
RUN echo "\nyes\n/mamba/miniforge3\n" | /bin/bash /tmp/Miniforge3-$(uname)-$(uname -m).sh
RUN rm /tmp/Miniforge3-$(uname)-$(uname -m).sh
ENV PATH="$PATH:/mamba/miniforge3/bin"
RUN mamba init

# add channels
RUN conda config --add channels conda-forge

RUN mamba update --all -y && mamba clean --all --force-pkgs-dirs -y

# copy repository
USER root
COPY . ${CODE_LOCATION}/
RUN rm -rf ${CODE_LOCATION}/.git
RUN chown -R ${USERNAME}:${USERNAME} ${CODE_LOCATION}


##########################
###### Naked Image #######
##########################

FROM base as build-0
ARG USERNAME=dev

# deactivate autoactivation of base image
USER ${USERNAME}
SHELL ["mamba", "run", "-n", "base", "/bin/bash", "-c"]
RUN conda config --set auto_activate_base false
SHELL ["/bin/bash", "-c"]

USER root
RUN mamba init
SHELL ["/mamba/miniforge3/bin/mamba", "run", "-n", "base", "/bin/bash", "-c"]
RUN conda config --set auto_activate_base false


##########################
###### Built Image #######
##########################

FROM base as build-1
ARG USERNAME=dev

# install environment
USER ${USERNAME}

RUN mamba env create -n benchmark -f ${CODE_LOCATION}/environment/environment.yml
SHELL ["mamba", "run", "-n", "benchmark", "/bin/bash", "-c"]
RUN pip install --extra-index-url https://pypi.nvidia.com --upgrade nvidia-dali-tf-plugin-cuda120
RUN pip install git+https://github.com/facebookresearch/pytorch3d.git@v0.7.7

# install the packages of the project
WORKDIR ${CODE_LOCATION}
RUN poetry install
WORKDIR /

SHELL ["/bin/bash", "-c"]
RUN mamba clean -aqy

# deactivate autoactivation of base image
SHELL ["mamba", "run", "-n", "base", "/bin/bash", "-c"]
RUN conda config --set auto_activate_base false
SHELL ["/bin/bash", "-c"]

USER root
RUN mamba init
SHELL ["/mamba/miniforge3/bin/mamba", "run", "-n", "base", "/bin/bash", "-c"]
RUN conda config --set auto_activate_base false
SHELL ["/bin/bash", "-c"]


##########################
##### Standard Image #####
##########################

FROM build-${build} as version-standard

ARG USERNAME=dev

WORKDIR ${CODE_LOCATION}

USER ${USERNAME}

CMD ["/bin/bash"]


##########################
###### Runai Image #######
##########################

FROM build-${build} as version-runai

ARG USERNAME=dev

USER root

SHELL ["/bin/sh", "-c"]

# change home directory to /myhome
RUN usermod -d /myhome ${USERNAME}

# install ssh
RUN apt update -y &&\
    apt install -y \
        openssh-server & \
    apt clean && \
    rm -rf /var/lib/apt/lists/*

# ssh configuration
RUN mkdir /var/run/sshd
RUN echo 'PermitRootLogin yes' >> /etc/ssh/sshd_config
RUN wget -O /etc/pam.d/sshd http://exampleconfig.com/static/raw/openssh/debian9/etc/pam.d/sshd
RUN sed 's@session\s*required\s*pam_loginuid.so@session optional pam_loginuid.so@g' -i /etc/pam.d/sshd

EXPOSE 22

# start ssh daemon
CMD ["/usr/sbin/sshd", "-D"]


##########################
###### Final Image #######
##########################

FROM version-${version} AS final