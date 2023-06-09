FROM python:latest

COPY requirements.txt /opt/requirements.txt

RUN pip3 install -r /opt/requirements.txt

COPY . /opt/server

WORKDIR /opt/server

ENV PYTHONPATH=/opt

CMD ["python", "app.py"]

EXPOSE 5001
