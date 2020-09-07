FROM python:3.7-slim

WORKDIR /app

ADD . /app

RUN pip install autofeat

ENTRYPOINT ["python"]

CMD ["/app/app.py"]