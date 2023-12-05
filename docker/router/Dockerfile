FROM python:3.11-slim-bullseye
RUN apt update
RUN apt install -y nano wget
RUN pip install grpcio-tools quart

# copy the contents of the current project to /workspace
COPY ../.. /workspace

# set the working directory to /workspace
WORKDIR /workspace

# generate the python files from the proto files
RUN cd /workspace/proto && bash ./generate.sh

CMD ["python", "./serving/router/router.py"]