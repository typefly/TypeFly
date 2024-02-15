from io import BytesIO
from PIL import Image
from typing import Optional, Tuple

import json, sys, os
import queue
import grpc
import asyncio

def image_to_bytes(image):
    # compress and convert the image to bytes
    imgByteArr = BytesIO()
    image.save(imgByteArr, format='WEBP')
    return imgByteArr.getvalue()

PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

sys.path.append(os.path.join(PARENT_DIR, "proto/generated"))
import hyrch_serving_pb2
import hyrch_serving_pb2_grpc

# VISION_SERVICE_IP = os.environ.get("VISION_SERVICE_IP", "localhost")
VISION_SERVICE_IP = "10.66.3.68"
YOLO_SERVICE_PORT = os.environ.get("YOLO_SERVICE_PORT", "50050").split(",")[0]

channel = grpc.insecure_channel(f'{VISION_SERVICE_IP}:8080')
stub = hyrch_serving_pb2_grpc.YoloServiceStub(channel)

detect_request = hyrch_serving_pb2.DetectRequest(image_data=image_to_bytes(Image.open("./images/kitchen.webp")))
response = stub.DetectStream(detect_request)

json_results = json.loads(response.json_data)
print(json_results)