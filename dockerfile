FROM python:3.6.2

WORKDIR /ImageRecognition/

COPY models /ImageRecognition/models
COPY templates /ImageRecognition/templates
COPY  application.properties Services.py requirements.txt /ImageRecognition/
Run pip install -r ./requirements.txt

COPY app.py __init__.py /ImageRecognition/


EXPOSE 5000

ENTRYPOINT python3 ./app.py