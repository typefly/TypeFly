import cv2
import requests
import json

def gstreamer_tet():
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

BASE_URL = "http://192.168.0.253:18080/control"

def test_control_api():
    # Test valid 'move' command
    # move_payload = {
    #     "command": "move",
    #     "dx": 0.3,
    #     "dy": 0.0,
    #     "body_frame": True,
    #     "timeout": 3.0
    # }
    # print("\nTesting 'move' command:")
    # response = requests.post(BASE_URL, json=move_payload)
    # print(f"Status: {response.status_code}")
    # print(f"Response: {response.json()}")

    # Test valid 'rotate' command
    rotate_payload = {
        "command": "rotate",
        "delta_angle": -720.0,
        "timeout": 1.0
    }
    print("\nTesting 'rotate' command:")
    response = requests.post(BASE_URL, json=rotate_payload)
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")

    # Test invalid command
    invalid_payload = {
        "command": "jump",  # Doesn't exist
        "dx": 1.0
    }
    print("\nTesting invalid command:")
    response = requests.post(BASE_URL, json=invalid_payload)
    print(f"Status: {response.status_code}")
    print(f"Response: {response.text}")

    # Test malformed JSON
    malformed_data = "{not_valid_json}"
    print("\nTesting malformed JSON:")
    response = requests.post(
        BASE_URL,
        data=malformed_data,
        headers={"Content-Type": "application/json"}
    )
    print(f"Status: {response.status_code}")
    print(f"Response: {response.text}")

if __name__ == "__main__":
    test_control_api()