FROM ultralytics/ultralytics:latest
RUN apt update
RUN pip install grpcio-tools quart uvicorn

# copy the contents of the current project to /workspace
COPY ../.. /workspace

# set the working directory to /workspace
WORKDIR /workspace

# generate the python files from the proto files
RUN cd /workspace/typefly/proto && bash ./generate.sh

CMD ["python", "typefly/serving/edge/run_edge.py"]