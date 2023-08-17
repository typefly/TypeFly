# build image
bash ./build.sh
# create a new container
docker run -td --privileged --net=host --ipc=host \
    --name="yolo-test" \
    yolo-test:0.1