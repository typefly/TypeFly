import os

PROJ_DIR = os.environ.get("PROJ_PATH", os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
EDGE_SERVICE_PORT = int(os.environ.get("EDGE_SERVICE_PORT", "50049"))

"""
The service information for the edge service.
The ports is a list of ports of the service.
the require_image is a boolean value indicating whether the service requires an image.
"""
SERVICE_INFO = {
    "yolo": {
        "host": "localhost",
        "port": [50050],
        "require_image": True,
    },
    # "clip": {
    #     "host": "localhost",
    #     "port": [50052],
    #     "require_image": True,
    # },
}


