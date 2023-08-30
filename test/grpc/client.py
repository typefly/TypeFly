import grpc
import yolo_pb2
import yolo_pb2_grpc
from PIL import Image
from io import BytesIO

def image_to_bytes(image):
    # compress and convert the image to bytes
    imgByteArr = BytesIO()
    image.save(imgByteArr, format='WEBP')
    return imgByteArr.getvalue()

def run():
    channel = grpc.insecure_channel('localhost:50051')
    stub = yolo_pb2_grpc.YoloServiceStub(channel)

    image_data = image_to_bytes(Image.open("../images/kitchen.webp"))

    print(len(image_data))
    
    response = stub.Detect(yolo_pb2.DetectRequest(image_data=image_data))
    print(f"Received: {response.results[0]}")

if __name__ == '__main__':
    run()