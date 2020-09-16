FROM python:3.8-slim-buster
ADD requirements.txt /home
RUN pip install -r /home/requirements.txt
ADD src/remote/ /golem/entrypoints/
VOLUME /golem/work /golem/output /golem/resource
