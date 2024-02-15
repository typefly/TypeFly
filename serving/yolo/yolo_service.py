import sys, os, gc
from concurrent import futures
from PIL import Image
from io import BytesIO
import json
import time
import grpc
import torch
from ultralytics import YOLO
import multiprocessing

PARENT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ROOT_PATH = os.environ.get("ROOT_PATH", PARENT_DIR)
SERVICE_PORT = os.environ.get("YOLO_SERVICE_PORT", "50050, 50051").split(",")

MODEL_PATH = os.path.join(ROOT_PATH, "./serving/yolo/models/")
MODEL_TYPE = "yolov8m.pt"

sys.path.append(ROOT_PATH)
sys.path.append(os.path.join(ROOT_PATH, "proto/generated"))
import hyrch_serving_pb2
import hyrch_serving_pb2_grpc

def load_model():
    model = YOLO(MODEL_PATH + MODEL_TYPE)
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device('cpu')
    model.to(device)
    print(f"GPU memory usage: {torch.cuda.memory_allocated()}")
    return model

def release_model(model):
    del model
    gc.collect()
    torch.cuda.empty_cache()

"""
    gRPC service class.
"""
class YoloService(hyrch_serving_pb2_grpc.YoloServiceServicer):
    def __init__(self, port):
        self.model = load_model()
        self.stream_mode = False
        self.port = port

    def reload_model(self):
        release_model(self.model)
        self.model = load_model()
    
    @staticmethod
    def bytes_to_image(image_bytes):
        return Image.open(BytesIO(image_bytes))
    
    @staticmethod
    def format_result(yolo_result):
        if yolo_result.probs is not None:
            print('Warning: Classify task do not support `tojson` yet.')
            return
        formatted_result = []
        data = yolo_result.boxes.data.cpu().tolist()
        h, w = yolo_result.orig_shape
        for i, row in enumerate(data):  # xyxy, track_id if tracking, conf, class_id
            box = {'x1': round(row[0] / w, 2), 'y1': round(row[1] / h, 2), 'x2': round(row[2] / w, 2), 'y2': round(row[3] / h, 2)}
            conf = row[-2]
            class_id = int(row[-1])

            name = yolo_result.names[class_id]
            if yolo_result.boxes.is_track:
                # result['track_id'] = int(row[-3])  # track ID
                name = f'{name}_{int(row[-3])}'
            result = {'name': name, 'confidence': round(conf, 2), 'box': box}
            
            if yolo_result.masks:
                x, y = yolo_result.masks.xy[i][:, 0], yolo_result.masks.xy[i][:, 1]  # numpy array
                result['segments'] = {'x': (x / w).tolist(), 'y': (y / h).tolist()}
            if yolo_result.keypoints is not None:
                x, y, visible = yolo_result.keypoints[i].data[0].cpu().unbind(dim=1)  # torch Tensor
                result['keypoints'] = {'x': (x / w).tolist(), 'y': (y / h).tolist(), 'visible': visible.tolist()}
            formatted_result.append(result)
        return formatted_result
    
    def process_image(self, image, id=None):
        if self.stream_mode:
            result = self.model.track(image, verbose=False, persist=True)[0]
        else:
            result = self.model(image, verbose=False)[0]
        result = {
            "image_id": id,
            "result": YoloService.format_result(result)
        }
        return json.dumps(result)

    def DetectStream(self, request, context):
        print(f"Received DetectStream request from {context.peer()} on port {self.port}, image_id: {request.image_id}")
        if not self.stream_mode:
            self.stream_mode = True
            self.reload_model()
        
        image = YoloService.bytes_to_image(request.image_data)
        return hyrch_serving_pb2.DetectResponse(json_data=self.process_image(image, request.image_id))
    
    def Detect(self, request, context):
        print(f"Received Detect request from {context.peer()} on port {self.port}, image_id: {request.image_id}")
        if self.stream_mode:
            self.stream_mode = False
            self.reload_model()

        image = YoloService.bytes_to_image(request.image_data)
        return hyrch_serving_pb2.DetectResponse(json_data=self.process_image(image, request.image_id))

def serve(port):
    print(f"Starting YoloService at port {port}")
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
    hyrch_serving_pb2_grpc.add_YoloServiceServicer_to_server(YoloService(port), server)
    server.add_insecure_port(f'[::]:{port}')
    server.start()
    server.wait_for_termination()

if __name__ == '__main__':
    # Create a pool of processes
    process_count = len(SERVICE_PORT)
    processes = []

    for i in range(process_count):
        process = multiprocessing.Process(target=serve, args=(SERVICE_PORT[i],))
        process.start()
        processes.append(process)

    # Wait for all processes to complete
    for process in processes:
        process.join()