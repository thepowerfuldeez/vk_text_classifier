FROM python:latest

ADD ./ ~/vk_text_classifier/
WORKDIR ~/vk_text_classifier/

RUN export LANG=C.UTF-8 && pip install pipenv
RUN pipenv install --system