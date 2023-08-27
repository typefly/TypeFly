FROM ultralytics/yolov5
RUN apt update
RUN apt install -y nano
RUN pip install Flask

COPY ./src ./src
ENV MAIN_PORT=50051
EXPOSE $MAIN_PORT

# CMD ["python", "./src/app.py"]