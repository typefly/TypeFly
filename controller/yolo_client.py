from io import BytesIO
from PIL import Image, ImageDraw, ImageFont

import grpc, json
import queue

import sys, cv2
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
        font = ImageFont.truetype('Monaco.ttf', 50)
        for result in results:
            draw.rectangle(((int(result['x1']), int(result['y1'])), (int(result['x2']), int(result['y2']))), fill=None, outline='blue', width=2)
            draw.text((int(result['x1']), int(result['y1']) - 10), result['label'], fill='red', font=font)
    
    def detect(self, image):
        image_bytes = YoloClient.image_to_bytes(image)
        request = hyrch_serving_pb2.DetectRequest(image_data=image_bytes)
        results = json.loads(self.stub.Detect(request).results)

        if self.yolo_results_queue is not None:
            while not self.yolo_results_queue.empty():
                self.yolo_results_queue.get()
            relative_results = []
            for result in results:
                relative_results.append({
                    'name': result['label'].replace(' ', '_'),
                    'x1': result['x1'] / image.width,
                    'y1': result['y1'] / image.height,
                    'x2': result['x2'] / image.width,
                    'y2': result['y2'] / image.height,
                })
            self.yolo_results_queue.put(relative_results)
        return results

if __name__ == "__main__":
    yolo_client = YoloClient()
    image = Image.open('../test/images/kitchen.webp')
    results = yolo_client.detect(image)
    YoloClient.plot_results(image, results)
    image.save('../test/images/kitchen_detected.webp')
    print(results)