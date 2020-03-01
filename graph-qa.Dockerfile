FROM pytorch/pytorch:1.3-cuda10.1-cudnn7-runtime

WORKDIR /graph-qa/

COPY requirements.txt .

EXPOSE 8091 8091
EXPOSE 8088 8088
EXPOSE 8080 8080
EXPOSE 8888 8888

RUN pip install -r requirements.txt
RUN python -m nltk.downloader punkt
