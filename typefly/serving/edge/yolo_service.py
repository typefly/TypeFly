import sys, os, gc
from concurrent import futures
from PIL import Image
from io import BytesIO
import json, time
import grpc
import torch
from ultralytics import YOLO

PROJ_DIR = os.environ.get("PROJ_PATH", os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
sys.path.append(os.path.join(PROJ_DIR, "typefly/proto"))
import hyrch_serving_pb2
import hyrch_serving_pb2_grpc

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models/")
MODEL_TYPE = "yolov8m.pt"

def load_model():
    model = YOLO(MODEL_PATH + MODEL_TYPE)
    if torch.cuda.is_available():
        model.to('cuda')
        print(f"GPU memory usage: {torch.cuda.memory_allocated()}")
    elif torch.backends.mps.is_available():
        model.to('mps')
        print(f"MPS memory usage: {torch.mps.current_allocated_memory()}")
    
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
        self.tracking_mode = False
        self.model = load_model()
        self.port = port

    def reload_model(self):
        if self.model is not None:
            release_model(self.model)
        self.model = load_model()

    @staticmethod
    def bytes_to_image(image_bytes) -> Image.Image:
        return Image.open(BytesIO(image_bytes))
    
    @staticmethod
    def format_result(yolo_result) -> list:
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

    def parse_request(self, request) -> tuple[Image.Image, dict]:
        info = json.loads(request.json_data)
        image = YoloService.bytes_to_image(request.image_data)

        if 'tracking_mode' not in info:
            info['tracking_mode'] = False

        if 'conf' not in info:
            info['conf'] = 0.3

        if self.tracking_mode != info['tracking_mode']:
            self.tracking_mode = info['tracking_mode']
            self.reload_model()

        return image, info
    
    def Detect(self, request, context) -> hyrch_serving_pb2.DetectResponse:
        # print(f"Received Detect request from {context.peer()} on port {self.port}")
        
        # start_time = time.time()
        image, info = self.parse_request(request)
        print(f"Received Detect request {info['image_id']}")

        if self.tracking_mode:
            yolo_result = self.model.track(image, verbose=False, conf=info['conf'], tracker="bytetrack.yaml")[0]
        else:
            yolo_result = self.model(image, verbose=False, conf=info['conf'])[0]

        info['result'] = YoloService.format_result(yolo_result)
        # print(f"Detection took {time.time() - start_time} seconds")
        return hyrch_serving_pb2.DetectResponse(json_data=json.dumps(info))

def serve(port, stop_event):
    print(f"Yolo service at port {port} [STARTING]")
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
    hyrch_serving_pb2_grpc.add_YoloServiceServicer_to_server(YoloService(port), server)
    server.add_insecure_port(f'[::]:{port}')
    server.start()

    try:
        # Wait until the event is set
        while not stop_event.is_set():
            time.sleep(1)
    except KeyboardInterrupt:
        print(f"YOLO service at port {port} [STOPPED]")
        server.stop(0)

if __name__ == "__main__":
    # test the service
    serve(50050)