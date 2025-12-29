import multiprocessing
import signal
import uvicorn
import os

from typefly.serving.yolo_service import serve as yolo_service
from typefly.serving.config import SERVICE_INFO, EDGE_SERVICE_PORT

def start_service(stop_event):
    processes = []
    for service_name, service_info in SERVICE_INFO.items():
        for port in service_info["port"]:
            if service_name == "yolo":
                process = multiprocessing.Process(target=yolo_service, args=(port, stop_event))
                process.start()
                processes.append(process)
            # elif other services, you can add more here
            else:
                raise ValueError(f"Unknown service: {service_name}")
    return processes

def main():
    stop_event = multiprocessing.Event()
    processes = start_service(stop_event)

    def cleanup(_signalnum, _frame):
        print("Shutting down YOLO services...")
        stop_event.set()
        for p in processes:
            p.join()
        print("Shutdown complete.")
        os._exit(0)

    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)
    from typefly.serving.gateway import app
    uvicorn.run(app, host="0.0.0.0", port=EDGE_SERVICE_PORT)

if __name__ == "__main__":
    main()
