import os

PROJ_DIR = os.environ.get("PROJ_PATH", os.path.dirname(os.path.abspath(__file__)))

SERVICE_INFO = [
    {"name": "yolo", "host": "localhost", "ports": [50050]},
    # {"name": "yolo3d", "host": "localhost", "ports": [50051]},
]
EDGE_SERVICE_PORT = int(os.environ.get("EDGE_SERVICE_PORT", "50049"))

