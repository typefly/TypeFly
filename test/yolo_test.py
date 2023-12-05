import sys
from PIL import Image
sys.path.append("..")
from controller.yolo_client import YoloClient

yolo_client = YoloClient()

image = Image.open("./images/kitchen.webp")
yolo_client.detect_local(image)