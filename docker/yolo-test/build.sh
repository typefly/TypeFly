# remove old container and image
docker stop -t 0 yolo-test
docker image rm -f yolo-test:0.1
docker rm -f yolo-test &>/dev/null

# build
docker build -t yolo-test:0.1 .