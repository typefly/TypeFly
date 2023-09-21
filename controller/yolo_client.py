from io import BytesIO
from PIL import Image, ImageDraw, ImageFont

import grpc, json
import queue

import sys, os
sys.path.append('../proto/generated')
import hyrch_serving_pb2
import hyrch_serving_pb2_grpc

# YOLO_SERVICE_IP = 'localhost'
YOLO_SERVICE_IP = '172.29.249.77'
YOLO_SERVICE_PORT = '50051'

class YoloClient():
    def __init__(self, yolo_results_queue: queue.Queue=None):
        channel = grpc.insecure_channel(f'{YOLO_SERVICE_IP}:{YOLO_SERVICE_PORT}')
        self.stub = hyrch_serving_pb2_grpc.YoloServiceStub(channel)
        self.yolo_results_queue = yolo_results_queue

    def image_to_bytes(image):
        # compress and convert the image to bytes
        imgByteArr = BytesIO()
        image.save(imgByteArr, format='WEBP')
        return imgByteArr.getvalue()
    
    def plot_results(frame, results):
        draw = ImageDraw.Draw(frame)
        for result in results:
            draw.rectangle(((int(result['x1']), int(result['y1'])), (int(result['x2']), int(result['y2']))), fill=None, outline='blue', width=2)
            draw.text((int(result['x1']), int(result['y1']) - 10), result['label'], fill='red')
    
    def detect(self, image):
        image_bytes = YoloClient.image_to_bytes(image)
        request = hyrch_serving_pb2.DetectRequest(image_data=image_bytes)
        results = json.loads(self.stub.Detect(request).results)

        if self.yolo_results_queue is not None:
            while not self.yolo_results_queue.empty():
                self.yolo_results_queue.get()
            self.yolo_results_queue.put(results)
        return results

if __name__ == "__main__":
    yolo_client = YoloClient()
    image = Image.open('../test/images/kitchen.webp')
    results = yolo_client.detect(image)
    YoloClient.plot_results(image, results)
    image.save('../test/images/kitchen_detected.webp')
    print(results)