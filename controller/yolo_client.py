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
        self.image_size = (640, 352)

    def image_to_bytes(image):
        # compress and convert the image to bytes
        imgByteArr = BytesIO()
        image.save(imgByteArr, format='WEBP')
        return imgByteArr.getvalue()
    
    def plot_results(frame, results):
        def str_float_to_int(value, multiplier):
            return int(float(value) * multiplier)
        draw = ImageDraw.Draw(frame)
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", size=50)
        w, h = frame.size
        for result in results:
            box = result["box"]
            draw.rectangle((str_float_to_int(box["x1"], w), str_float_to_int(box["y1"], h), str_float_to_int(box["x2"], w), str_float_to_int(box["y2"], h)),
                        fill=None, outline='blue', width=2)
            draw.text((str_float_to_int(box["x1"], w), str_float_to_int(box["y1"], h) - 10), result["name"], fill='red', font=font)
    
    def detect(self, image):
        image_bytes = YoloClient.image_to_bytes(image.resize(self.image_size))
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