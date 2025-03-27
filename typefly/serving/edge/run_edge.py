import multiprocessing
import signal
import uvicorn
import os, sys

PROJ_DIR = os.environ.get("PROJ_PATH", os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
sys.path.insert(0, PROJ_DIR)
from typefly.serving.edge import yolo_service
YOLO_SERVICE_INFO = { "host": "localhost", "port" : [50050, 50051] }
EDGE_SERVICE_PORT = int(os.environ.get("EDGE_SERVICE_PORT", "50049"))

def start_yolo_service(stop_event):
    processes = []
    for port in YOLO_SERVICE_INFO["port"]:
        process = multiprocessing.Process(target=yolo_service.serve, args=(port, stop_event))
        process.start()
        processes.append(process)
    return processes

def main():
    stop_event = multiprocessing.Event()
    processes = start_yolo_service(stop_event)

    def cleanup(_signalnum, _frame):
        print("Shutting down YOLO services...")
        stop_event.set()
        for p in processes:
            p.join()
        print("Shutdown complete.")
        os._exit(0)

    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)

    uvicorn.run("typefly.serving.edge.edge_service:app", host="0.0.0.0", port=EDGE_SERVICE_PORT, reload=False)

if __name__ == "__main__":
    main()
