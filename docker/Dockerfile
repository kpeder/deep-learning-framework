FROM python:3.10-bookworm

RUN apt-get update -y \
 && apt-get install -y build-essential git pre-commit python3-pip python3-dev \
 && apt-get install -y libbz2-dev libffi-dev liblzma-dev libsnappy-dev libsqlite3-dev libssl-dev

COPY ../. /app/deep-learning-framework

WORKDIR /app/deep-learning-framework
RUN make install
