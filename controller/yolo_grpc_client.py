from io import BytesIO
from PIL import Image
from typing import Optional, Tuple

import json, sys, os
import queue
import grpc
import asyncio

from .yolo_client import SharedYoloResult

PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

sys.path.append(os.path.join(PARENT_DIR, "proto/generated"))
import hyrch_serving_pb2
import hyrch_serving_pb2_grpc

TYPEFLY_SERVICE_IP = os.environ.get("TYPEFLY_SERVICE_IP", "localhost")
YOLO_SERVICE_PORT = os.environ.get("YOLO_SERVICE_PORT", "50050").split(",")[0]

'''
Access the YOLO service through gRPC.
'''
class YoloGRPCClient():
    def __init__(self, is_local_service=False, shared_yolo_result: SharedYoloResult=None):
        self.is_local_service = is_local_service
        if is_local_service:
            channel = grpc.insecure_channel(f'localhost:{YOLO_SERVICE_PORT}')
        else:
            channel = grpc.aio.insecure_channel(f'{TYPEFLY_SERVICE_IP}:{YOLO_SERVICE_PORT}')
        self.stub = hyrch_serving_pb2_grpc.YoloServiceStub(channel)
        self.image_size = (640, 352)
        self.image_queue = queue.Queue()
        self.shared_yolo_result = shared_yolo_result
        self.latest_result_with_image = None
        if not is_local_service:
            self.latest_result_with_image_lock = asyncio.Lock()
            self.image_id_lock = asyncio.Lock()
            self.image_id = 0

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
        if self.shared_yolo_result is not None:
            self.shared_yolo_result.set(json_results)

    async def detect(self, image):
        if self.is_local_service:
            self.detect_local(image)
            return
        image_bytes = YoloGRPCClient.image_to_bytes(image.resize(self.image_size))

        async with self.image_id_lock:
            image_id = self.image_id
            self.image_queue.put((self.image_id, image))
            self.image_id += 1

        detect_request = hyrch_serving_pb2.DetectRequest(image_id=image_id, image_data=image_bytes)
        response = await self.stub.DetectStream(detect_request)

        async with self.latest_result_with_image_lock:
            json_results = json.loads(response.json_data)
            if self.image_queue.empty():
                return
            # discard old images
            while self.image_queue.queue[0][0] < json_results['image_id']:
                self.image_queue.get()
            # discard old results
            if self.image_queue.queue[0][0] > json_results['image_id']:
                return
            self.latest_result_with_image = (self.image_queue.get()[1], json_results)
            if self.shared_yolo_result is not None:
                self.shared_yolo_result.set(self.latest_result_with_image)