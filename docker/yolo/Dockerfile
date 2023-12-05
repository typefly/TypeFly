FROM ultralytics/yolov5
RUN apt update
RUN apt install -y nano wget
RUN pip install grpcio-tools ultralytics

# copy the contents of the current project to /workspace
COPY ../.. /workspace

# set the working directory to /workspace
WORKDIR /workspace

# generate the python files from the proto files
RUN cd /workspace/proto && bash ./generate.sh

CMD ["python", "./serving/yolo/yolo_service.py"]