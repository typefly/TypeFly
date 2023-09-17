from io import BytesIO
from PIL import Image, ImageDraw

import grpc

import sys, os
sys.path.append('./proto/generated')
import yolo_pb2
import yolo_pb2_grpc

LOCAL_SERVICE_IP = os.environ.get('SERVICE_IP', 'localhost')

class YoloClient():
    def __init__(self):
        channel = grpc.insecure_channel(f'{LOCAL_SERVICE_IP}:50051')
        self.stub = yolo_pb2_grpc.YoloServiceStub(channel)

    def image_to_bytes(image):
        # compress and convert the image to bytes
        imgByteArr = BytesIO()
        image.save(imgByteArr, format='WEBP')
        return imgByteArr.getvalue()
    
    def plot_results(frame, results):
        draw = ImageDraw.Draw(frame)
        for result in results:
            draw.rectangle(((int(result.x1), int(result.y1)), (int(result.x2), int(result.y2))), fill=None, outline='blue', width=2)
            draw.text((int(result.x1), int(result.y1) - 10), result.label, fill='red')
    
    # def cv2_to_pil(image):
    #     return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    def detect(self, image):
        image_bytes = YoloClient.image_to_bytes(image)
        request = yolo_pb2.DetectRequest(image_data=image_bytes)
        return self.stub.Detect(request)
    
if __name__ == "__main__":
    yolo_client = YoloClient()
    image = Image.open('../test/images/kitchen.webp')
    results = yolo_client.detect(image).results
    YoloClient.plot_results(image, results)
    image.save('../test/images/kitchen_detected.webp')
    print(results)