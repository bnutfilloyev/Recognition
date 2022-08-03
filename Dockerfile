FROM python:3.9.2

ENV PYTHONUNBUFFERED 1

WORKDIR /app

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY . ./
ENV PYTHONPATH app

CMD ["python", "main.py"]