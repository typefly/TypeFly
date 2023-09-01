import grpc
import yolo_pb2
import yolo_pb2_grpc
from PIL import Image, ImageDraw
from io import BytesIO

import cv2, os
import numpy as np

SERVICE_IP = os.environ.get('SERVICE_IP', 'localhost')

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

def cv2_to_pil(image):
    return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

def run():
    channel = grpc.insecure_channel(SERVICE_IP + ':50051')
    stub = yolo_pb2_grpc.YoloServiceStub(channel)

    # image = Image.open("../images/kitchen.webp")
    # image_data = image_to_bytes(image)
    
    # response = stub.Detect(yolo_pb2.DetectRequest(image_data=image_data))

    # print(f"Received: {response.results[0]}")
    # plot_results(image, response.results)
    # image.show()

    cap = cv2.VideoCapture(1)
    # Check if camera opened successfully
    if (cap.isOpened() == False):
        print("Unable to read camera feed")
    
    while True:
        ret, frame = cap.read()
        if ret == True:
            image = cv2_to_pil(frame)
            image_data = image_to_bytes(image)
            response = stub.Detect(yolo_pb2.DetectRequest(image_data=image_data))
            plot_results(image, response.results)
            cv2.imshow('Frame', cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR))
            # Press Q on keyboard to exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

if __name__ == '__main__':
    run()