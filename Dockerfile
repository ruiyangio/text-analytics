FROM ubuntu:16.04
LABEL maintainer Rui "ruiyangwind@gmail.com"
RUN apt-get update -y
RUN apt-get install -y python-pip python-dev build-essential
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
EXPOSE 3020
ENTRYPOINT ["python3"]
CMD ["app.py"]