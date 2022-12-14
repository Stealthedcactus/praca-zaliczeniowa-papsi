FROM python:3.10-slim-bullseye

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

WORKDIR /app

COPY requirements.txt requirements.txt
COPY /app /app

RUN pip install --no-cache-dir --upgrade -r requirements.txt

