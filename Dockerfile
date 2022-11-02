FROM python:3.11

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

WORKDIR /app/

COPY ./app /app
RUN pip install --no-cache-dir -r requirements.txt


CMD["python", "./app/main.py"]