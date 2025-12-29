import multiprocessing
import signal
import uvicorn
import os

from .yolo_service import serve as yolo_service
from .config import SERVICE_INFO, EDGE_SERVICE_PORT

def start_yolo_service(stop_event):
    processes = []
    for service in SERVICE_INFO:
        for port in service["ports"]:
            if service["name"] == "yolo":
                process = multiprocessing.Process(target=yolo_service, args=(port, stop_event))
                process.start()
                processes.append(process)
            else:
                raise ValueError(f"Unknown service: {service['name']}")
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
    from .gateway import app
    uvicorn.run(app, host="0.0.0.0", port=EDGE_SERVICE_PORT)

if __name__ == "__main__":
    main()
