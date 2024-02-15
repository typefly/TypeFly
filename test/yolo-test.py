import sys
from PIL import Image
sys.path.append("..")
from controller.yolo_grpc_client import YoloGRPCClient
from controller.yolo_client import SharedYoloResult

syr = SharedYoloResult()
yolo_client = YoloGRPCClient(False, syr)

image = Image.open("./images/kitchen.webp")
yolo_client.detect_local(image)
print(syr.get())