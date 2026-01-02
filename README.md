# TypeFly
TypeFly aims to provide an easy platform for developping robot control system with large language models (LLMs). Link to our [full Paper](https://drive.google.com/file/d/1COrozqEIk6v8DLxI3vCgoSUEWpnsc2mu/view) and [webpage](https://typefly.github.io/).

Also, check out the demo video here: [Demo 1: Find edible or drinkable items](http://www.youtube.com/watch?v=HEJYaTLWKfY), [Demo 2: Find a specific chair](http://www.youtube.com/watch?v=QwnBniFaINE).

## 1. Installation
[Optional] Create a conda environment.
```bash
conda create -n typefly python=3.12
conda activate typefly
```

Clone this repo and install the package.
```bash
git clone https://github.com/typefly/TypeFly.git
cd TypeFly
pip install -e .
```

## 2. Hardware Requirement
Editing `typefly/config/robot_info.json` for different robot setups.

### Test without Robot
By default, typefly will try to access your camera with `cv2.VideoCapture(0)` and plan with that visual capture. This is for you to quickly try out the planning function.

### Tello Drone
TypeFly works with the DJI Tello drone. However, since Tello drone requires your device to connect to its WiFi network and TypeFly requires an Internet connection for LLM access, you need to have both WiFi adapter and ethernet adapter to run TypeFly for tello. To use this option, change the `robot_type` from `virtual` to `tello`.

### Go2 Dog
To control a Unitree Go2 robot dog with TypeFly, you need to install ROS2 and run the [go2_ros2_sdk](https://github.com/abizovnuralem/go2_ros2_sdk).

### Other Robots
To support other robots, you need to implement the robot control interface based on the `RobotWrapper`, see examples in `typefly/platforms/*`.

## 3. OPENAI API KEY Requirement
TypeFly use GPT API as the remote LLM planner, please make sure you have set the `OPENAI_API_KEY` environment variable.

## 4. Setup Vision Encoder
### Local Service
TypeFly uses YOLO to generate the scene description. We provide a scalable implementation of the http yolo service. Enter this to run the service directly on your machine.
```bash
cd typefly/proto && bash generate.sh
python -m typefly.serving
```

### Docker (Optional)
We recommand using [docker](https://docs.docker.com/engine/install/ubuntu/) to run the YOLO and the http router. To deploy the YOLO servive with docker, please install the [Nvidia Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html), then run the following command:
```bash
make serving_build
```

## 5. Start TypeFly Web UI
To play with the TypeFly, please run the following command after setting up the vision service:
```bash
python -m typefly.webui
```
This will start the web UI at `http://localhost:50000`. You should be able to see the image capture window displayed with YOLO detection results. You can test the planning ability of TypeFly by typing in the chat box. (If your vision service is on a different machine (e.g. an edge server or cloud), you need to setup the `EDGE_SERVICE_IP` and `EDGE_SERVICE_PORT` environment variables.)
