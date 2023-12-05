# TypeFly
## Hardware Requirement
TypeFly works with DJI Tello drone by default. To support other drones, you need to implement the `DroneWrapper` interface in `controller/abs/drone_wrapper.py`.

## Vision Encoder
TypeFly uses YOLOv8 to generate the scene description. We provide the implementation of gRPC YOLO service and a optional http router to serve as a scheduler when working with multiple drones. We recommand using docker to run the YOLO and router. To deploy the YOLO servive, please run the following command:
```bash
make SERVICE=yolo build
```
Optional: To deploy the router, please run the following command:
```bash
make SERVICE=router build
```

## TypeFly Web UI
To play with the TypeFly web UI, please run the following command:
```bash
make typefly
```
This will start the web UI at `http://localhost:50001` with your default camera (please make sure your device has a camera) and a virtual drone wrapper. You should be able to see the image capture window displayed with YOLO detection results. You can test the planning ability of TypeFly by typing in the chat box. 

To work with a real drone, please disable the `--use_virtual_cam` flag in `Makefile`.

## Task Execution
Here are some examples of task descriptions, the `[Q]` prefix indicates TypeFly will output an answer to the question:
- `Can you find something edible?`
- `Can you see a person behind you?`
- `[Q] Tell me how many people you can see?`