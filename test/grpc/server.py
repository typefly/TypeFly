import grpc
from concurrent import futures
import yolo_pb2
import yolo_pb2_grpc
from PIL import Image
from io import BytesIO

import torch, threading

MODEL_SOURCE = "ultralytics/yolov5"
MODEL_NAME = "yolov5s"

class YoloService(yolo_pb2_grpc.YoloServiceServicer):
    def __init__(self):
        self.local_storage = threading.local()

    def get_model(self):
        self.load_model()
        return self.local_storage.model
    
    def load_model(self):
        if not hasattr(self.local_storage, "model"):
            self.local_storage.model = torch.hub.load(MODEL_SOURCE, MODEL_NAME)

    def format_results(self, results):
        return results.pandas().xyxy[0]

    def Detect(self, request, context):
        image = Image.open(BytesIO(request.image_data))
        local_model = self.get_model()
        results = local_model(image).pandas().xyxy[0].values.tolist()
        detect_response = [yolo_pb2.DetectedObject(
                x1=result[0],
                y1=result[1],
                x2=result[2],
                y2=result[3],
                confidence=result[4],
                class_=result[5],
                label=result[6]
            )
            for result in results
        ]
        return yolo_pb2.DetectResponse(results=detect_response)

def prewarm_threads(num_threads, yolo_service_instance):
    """Pre-warm threads by loading YOLO models."""
    threads = []

    def worker():
        yolo_service_instance.load_model()

    for _ in range(num_threads):
        thread = threading.Thread(target=worker)
        threads.append(thread)
        thread.start()

    # Wait for all threads to complete
    for thread in threads:
        thread.join()

def serve(yolo_service=YoloService()):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    yolo_pb2_grpc.add_YoloServiceServicer_to_server(yolo_service, server)
    server.add_insecure_port('[::]:50051')
    server.start()
    server.wait_for_termination()

if __name__ == '__main__':
    yolo_service = YoloService()
    prewarm_threads(2, yolo_service)
    serve(yolo_service)