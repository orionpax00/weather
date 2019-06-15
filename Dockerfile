FROM python:3.7

ENV PYTHONUNBUFFERED 1

RUN mkdir /regpip

WORKDIR /regpip

ADD . /regpip/

RUN pip install -r requirements.txt
