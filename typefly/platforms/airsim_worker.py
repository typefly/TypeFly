import airsim
import numpy as np
import time
from multiprocessing import Queue

def airsim_worker(cmd_queue: Queue, img_queue: Queue, vehicle_name: str, camera_name: str, host: str):
    import tornado.ioloop
    import tornado.gen

    client = airsim.MultirotorClient(ip=host)
    client.confirmConnection()
    client.enableApiControl(True, vehicle_name)
    client.armDisarm(True, vehicle_name)
    print(f"[AirSim Worker] ✅ متصل!")

    # دالة الكاميرا - تشتغل في thread منفصل عن الأوامر
    import threading

    stop_event = threading.Event()

    def camera_loop():
        while not stop_event.is_set():
            try:
                responses = client.simGetImages([
                    airsim.ImageRequest(
                        camera_name,
                        airsim.ImageType.Scene,
                        False, False
                    )
                ], vehicle_name=vehicle_name)
                if responses and responses[0].width > 0:
                    r = responses[0]
                    raw = np.frombuffer(r.image_data_uint8, dtype=np.uint8)
                    frame = raw.reshape(r.height, r.width, 3)
                    if not img_queue.full():
                        try:
                            img_queue.put_nowait(frame.copy())
                        except Exception:
                            pass
            except Exception as e:
                pass
            time.sleep(1.0 / 10.0)  # 10 FPS للكاميرا

    cam_thread = threading.Thread(target=camera_loop, daemon=True)
    cam_thread.start()

    # حلقة الأوامر الرئيسية
    while True:
        try:
            if not cmd_queue.empty():
                cmd = cmd_queue.get_nowait()

                if cmd['type'] == 'move':
                    dx = float(cmd['dx'])
                    dy = float(cmd['dy'])
                    duration = float(cmd['duration'])
                    try:
                        client.moveByVelocityAsync(
                            dx, dy, 0.0, duration,
                            vehicle_name=vehicle_name
                        ).join()
                        print(f"[AirSim Worker] ✅ تحرك: dx={dx}, dy={dy}, {duration}s")
                    except Exception as e:
                        print(f"[AirSim Worker] ❌ حركة فشلت: {e}")

                elif cmd['type'] == 'rotate':
                    deg = float(cmd['deg'])
                    try:
                        client.rotateByYawRateAsync(
                            deg, 1.0,
                            vehicle_name=vehicle_name
                        ).join()
                        print(f"[AirSim Worker] ✅ دوران: {deg}°")
                    except Exception as e:
                        print(f"[AirSim Worker] ❌ دوران فشل: {e}")

                elif cmd['type'] == 'stop':
                    stop_event.set()
                    break

        except Exception as e:
            pass

        time.sleep(0.05)