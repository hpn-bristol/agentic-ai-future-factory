FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y python3 python3-pip
RUN mkdir /app
COPY ./requirements.txt /app/requirements.txt
RUN pip3 install --no-cache-dir --upgrade -r ./app/requirements.txt

COPY . /app/
WORKDIR /app/

ENV PYTHONPATH=/app
RUN ls -lah /app/*

RUN chmod +x ./start.sh

EXPOSE 8989
CMD ["./start.sh"]
