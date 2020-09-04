FROM python:3.8-slim-buster
COPY src/run.sh src/search.py /golem/entrypoints/
VOLUME /golem/work /golem/output /golem/resource
