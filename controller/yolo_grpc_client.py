from io import BytesIO
from PIL import Image
from typing import Optional, Tuple

import json, sys
import queue
import grpc
import asyncio

from yolo_client import SharedYoloResults

sys.path.append("../proto/generated")
import hyrch_serving_pb2
import hyrch_serving_pb2_grpc

# YOLO_SERVICE_IP = 'localhost'
YOLO_SERVICE_IP = '172.29.249.77'
YOLO_SERVICE_PORT = '50050'
ROUTER_SERVICE_PORT = '50049'

'''
Access the YOLO service through gRPC.
'''
class YoloGRPCClient():
    def __init__(self, is_local_service=False, shared_yolo_results: SharedYoloResults=None):
        self.is_local_service = is_local_service
        if is_local_service:
            channel = grpc.insecure_channel(f'{YOLO_SERVICE_IP}:{YOLO_SERVICE_PORT}')
        else:
            channel = grpc.aio.insecure_channel(f'{YOLO_SERVICE_IP}:{YOLO_SERVICE_PORT}')
        self.stub = hyrch_serving_pb2_grpc.YoloServiceStub(channel)
        self.image_size = (640, 352)
        self.image_queue = queue.Queue()
        self.shared_yolo_results = shared_yolo_results
        self.latest_result_with_image = None
        if not is_local_service:
            self.latest_result_with_image_lock = asyncio.Lock()

    def local_service(self):
        return self.is_local_service

    def image_to_bytes(image):
        # compress and convert the image to bytes
        imgByteArr = BytesIO()
        image.save(imgByteArr, format='WEBP')
        return imgByteArr.getvalue()

    def retrieve(self) -> Optional[Tuple[Image.Image, str]]:
        return self.latest_result_with_image
    
    def detect_local(self, image):
        image_bytes = YoloGRPCClient.image_to_bytes(image.resize(self.image_size))
        self.image_queue.put(image)

        detect_request = hyrch_serving_pb2.DetectRequest(image_data=image_bytes)
        response = self.stub.DetectStream(detect_request)
        
        json_results = json.loads(response.json_data)
        self.latest_result_with_image = (self.image_queue.get(), json_results)
        if self.shared_yolo_results is not None:
            self.shared_yolo_results.set(json_results)

    async def detect(self, image):
        image_bytes = YoloGRPCClient.image_to_bytes(image.resize(self.image_size))
        self.image_queue.put(image)

        detect_request = hyrch_serving_pb2.DetectRequest(image_data=image_bytes)
        response = await self.stub.DetectStream(detect_request)

        async with self.latest_result_with_image_lock:
            json_results = json.loads(response.json_data)
            self.latest_result_with_image = (self.image_queue.get(), json_results)
            if self.shared_yolo_results is not None:
                self.shared_yolo_results.set(json_results)