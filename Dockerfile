FROM python:3.8-slim-buster
RUN pip install pydicom
COPY src/remote/run src/remote/search.py /golem/entrypoints/
VOLUME /golem/work /golem/output /golem/resource
