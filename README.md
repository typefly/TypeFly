# TypeFly
TypeFly aims to provide an easy platform for developping robot control system with large language models (LLMs). Link to our [full Paper](https://www.computer.org/csdl/journal/tm/2025/09/10970379/260Skt3CSnS) and [webpage](https://typefly.github.io/).

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

### Petoi Quadruped
TypeFly works with Petoi quadrupeds (Bittle / Nybble / Cub) running the [OpenCatESP32 firmware](https://github.com/Leonana69/typego-petoi-firmware). The Petoi is driven over plain HTTP and uses **two boards**: the OpenCatESP32 control board (locomotion and body pose, JSON API on port `80`) and a separate Seeed XIAO ESP32S3 camera board running `esp32-xiao-cam-stream`, which serves an MJPEG video stream for YOLO. To use it, set `robot_type` to `petoi` and add both board addresses to `extra` in `typefly/config/robot_info.json`:
```json
{
    "robot_id": "petoi1",
    "robot_type": "petoi",
    "extra": {
        "ip": "192.168.1.50",
        "camera_ip": "192.168.1.51",
        "walk_speed": 0.07,
        "rotate_speed": 45.0
    }
}
```
- `ip` (required): the OpenCatESP32 control board address.
- `camera_ip` (required for vision): the XIAO camera board address.
- `walk_speed` (optional, m/s) and `rotate_speed` (optional, deg/s): movement is open-loop (timed gaits), so calibrate these for your surface.

Beyond the common movement and vision skills, the Petoi exposes a few expressive body-pose skills driven by the firmware's `euler` command: `nod` (a "yes"), `shake_head` (a "no"), and `look_object` (track an object with the head for a few seconds).

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
