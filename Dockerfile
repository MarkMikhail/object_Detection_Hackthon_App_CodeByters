FROM ubuntu:16.04
ENV http_proxy $HTTP_PROXY
ENV https_proxy $HTTPS_PROXY
ARG DOWNLOAD_LINK=http://registrationcenter-download.intel.com/akdlm/irc_nas/16612/l_openvino_toolkit_p_2020.2.120.tgz
ARG INSTALL_DIR=/opt/intel/openvino
ARG TEMP_DIR=/tmp/openvino_installer
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    cpio \
    sudo \
    lsb-release && \
    rm -rf /var/lib/apt/lists/*
RUN mkdir -p $TEMP_DIR && cd $TEMP_DIR && \
    wget -c $DOWNLOAD_LINK && \
    tar xf l_openvino_toolkit*.tgz && \
    cd l_openvino_toolkit* && \
    sed -i 's/decline/accept/g' silent.cfg && \
    ./install.sh -s silent.cfg && \
    rm -rf $TEMP_DIR
RUN $INSTALL_DIR/install_dependencies/install_openvino_dependencies.sh
# build Inference Engine samples
RUN apt install libpq-dev python3-dev -y
RUN apt install software-properties-common -y
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt update
RUN apt install python3.7 -y
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.7 1

#RUN pip3 install --upgrade setuptools
RUN /bin/bash -c "source $INSTALL_DIR/bin/setupvars.sh"
RUN ./$INSTALL_DIR/deployment_tools/demo/demo_squeezenet_download_convert_run.sh
#RUN $INSTALL_DIR/deployment_tools/model_optimizer/install_prerequisites/install_prerequisites.sh

#RUN mkdir $INSTALL_DIR/deployment_tools/inference_engine/samples/build && cd $INSTALL_DIR/deployment_tools/inference_engine/samples/build && \
   # /bin/bash -c "source $INSTALL_DIR/bin/setupvars.sh && cmake .. && make -j1"

# installaing prerequisites
#RUN adduser  aniket
#WORKDIR /home/aniket
#USER aniket

#ENV PATH="/home/aniket/.local/bin:${PATH}"
#RUN cd  $INSTALL_DIR/deployment_tools/model_optimizer/install_prerequisites
#RUN ./$INSTALL_DIR/deployment_tools/model_optimizer/install_prerequisites/install_prerequisites.sh
#RUN ./$INSTALL_DIR/deployment_tools/demo/demo_squeezenet_download_convert_run.sh
# RUN ./$INSTALL_DIR/deployment_tools/demo/demo_security_barrier_camera.sh -y

#RUN apt-get update -y
RUN apt-get update
RUN apt-get install pciutils wget sudo kmod curl lsb-release cpio udev python3-pip libcanberra-gtk3-module git -y

RUN pip3 install --upgrade pip

RUN pip3 install opencv-python
RUN pip3 install requests
RUN pip3 install pyyaml

#COPY . /
#Install smart video workshop dependencies 
#RUN apt-get install libgflags-dev -y 
RUN pip3 install cogapp
#RUN chown  aniket:aniket -R root/inference_engine_samples_build/
#RUN cd root/inference_engine_samples_build && make
#RUN chown  aniket:aniket -R root/inference_engine_demos_build/
#RUN cd root/inference_engine_demos_build && make
RUN cd $INSTALL_DIR/deployment_tools/tools/model_downloader
#RUN python3 requirements.in
#RUN python3 $INSTALL_DIR/deployment_tools/tools/model_downloader/downloader.py --name mobilenet-ssd,person-detection-retail-0013
RUN apt-get install mosquitto mosquitto-clients -y
RUN pip3 install numpy paho-mqtt jupyter
#azure python sdk dependensies
#Azure python sdk 
##FROM amd64/python:3.7-slim-buster

##WORKDIR /app


#RUN apt-get install -y build-essential python3.6 python3.6-dev python3-pip python3.6-venv
# RUN apt-get install -y git

# update pip
RUN apt-get update && \
    apt-get install -y software-properties-common && \
    rm -rf /var/lib/apt/lists/*

# RUN sudo add-apt-repository ppa:deadsnakes/ppa
RUN apt-get update
RUN apt-get install python3.6 -y
RUN sudo apt-get install libpython3.6
RUN sudo apt-get install libpython3.7
#RUN wget https://bootstrap.pypa.io/get-pip.py
#RUN python3.6 get-pip.py
#RUN python3.6 -m pip install https://cntk.ai/PythonWheel/CPU-Only/cntk-2.1-cp36-cp36m-linux_x86_64.whl
#COPY requirements.txt ./
#RUN pip install -r requirements.txt	


RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.6 1
RUN pip3 install -U numpy 
RUN apt-get install -y python3.6-dev
RUN pip3 install paho-mqtt
RUN pip3 install requests
RUN pip3 install pyyaml
RUN pip3 install azure-iot-device~=2.0.0
RUN sudo apt-get install libssl-dev -y
RUN pip3 install opencv-python==4.1.2.30
RUN /bin/bash -c "source $INSTALL_DIR/bin/setupvars.sh"
#RUN pip3 install scikit-image
RUN pip3 install scipy
#COPY . .


#for interactice face detcction
COPY main_app/ main_app/

RUN cd main_app/
COPY run.sh run.sh
RUN chmod 743 run.sh
#Final run 
ADD "https://www.random.org/cgi-bin/randbyte?nbytes=10&format=h" skipcache
CMD ./run.sh



