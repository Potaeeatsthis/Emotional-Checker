FROM docker.io/tensorflow/tensorflow:2.11.0-gpu
RUN apt-get install git vim
RUN pip3 install tensorflow==2.12.0 numpy logging boto3 zipfile pillow matplotlib
