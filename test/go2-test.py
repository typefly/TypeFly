import cv2
import numpy as np

pipeline_str = """
            udpsrc address=230.1.1.1 port=1720 multicast-iface=wlan0
            ! application/x-rtp, media=video, encoding-name=H264
            ! rtph264depay
            ! h264parse
            ! avdec_h264
            ! videoconvert
            ! video/x-raw, format=BGR
            ! appsink name=appsink emit-signals=true max-buffers=1 drop=true
        """
cap = cv2.VideoCapture(pipeline_str, cv2.CAP_GSTREAMER)
if not cap.isOpened():
    raise RuntimeError("Failed to open GStreamer pipeline")

frame_rate = cap.get(cv2.CAP_PROP_FPS)
print(f"Frame rate: {frame_rate} FPS")
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to read frame")
        break

    # Process the frame (for example, display it)
    cv2.imshow('Frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()