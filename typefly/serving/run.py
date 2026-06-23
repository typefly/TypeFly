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
    # Bind to localhost by default; opt in to a public interface explicitly
    # (e.g. EDGE_SERVICE_HOST=0.0.0.0) once auth/rate-limiting are in place (P4).
    host = os.environ.get("EDGE_SERVICE_HOST", "127.0.0.1")
    # access_log=False hides the per-request "POST /process 200 OK" lines (one per
    # frame, ~10/s) while keeping startup/warning logs.
    uvicorn.run(app, host=host, port=EDGE_SERVICE_PORT, access_log=False)

if __name__ == "__main__":
    main()
