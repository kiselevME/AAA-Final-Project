FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1

RUN mkdir /app

COPY ./requirements.txt ./requirements-dev.txt /app/

WORKDIR /app

RUN python -m pip install --no-cache-dir -r requirements.txt -r requirements-dev.txt

COPY . /app

EXPOSE 8080
CMD ["python", "run.py"]
