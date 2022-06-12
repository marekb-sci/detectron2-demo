FROM continuumio/miniconda3:latest

WORKDIR /home/detectron
SHELL ["/bin/bash", "-c"]

#instal gcc
RUN apt update \
	&& apt install -y build-essential \ 
	&& apt-get install ffmpeg libsm6 libxext6  -y

#install packages 
RUN conda install pytorch torchvision torchaudio cpuonly -c pytorch \
	&& conda install -c conda-forge opencv \
        && pip install git+https://github.com/facebookresearch/detectron2.git \
	&& conda install -c anaconda flask

#COPY models models
COPY download_models.sh detection_tools.py main.py ./ 
ADD templates ./templates
RUN chmod +x download_models.sh && ./download_models.sh

EXPOSE 5000

CMD ["python", "main.py"] 
#["./boot.sh"]
