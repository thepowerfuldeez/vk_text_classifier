FROM python:latest

COPY ./ ~/vk_text_classifier/
WORKDIR ~/vk_text_classifier/

RUN export LANG=C.UTF-8 && python3.6 -m pip install -r ./requirements.txt