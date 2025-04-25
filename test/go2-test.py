import cv2
import requests

def gstreamer_test(folder_path: str = None):
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
    frame_index = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame")
            break

        # Process the frame (for example, display it)
        cv2.imshow('Frame', frame)
        if folder_path:
            cv2.imwrite(folder_path + f"/frame_{frame_index:04}.jpg", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        frame_index += 1
    cap.release()
    cv2.destroyAllWindows()

BASE_URL = "http://192.168.8.213:18080"
def test_move(dx, dy):
    move_payload = {
        "command": "move",
        "dx": dx,
        "dy": dy,
        "body_frame": True,
        "timeout": 3.0
    }
    print("\nTesting 'move' command:")
    response = requests.post(BASE_URL + '/control', json=move_payload)
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")

def test_rotate(delta_angle):
    # Convert degrees to radians
    delta_rad = delta_angle * (3.14159 / 180)
    rotate_payload = {
        "command": "rotate",
        "delta_rad": delta_rad,
        "timeout": 1.0
    }
    print("\nTesting 'rotate' command:")
    response = requests.post(BASE_URL + '/control', json=rotate_payload)
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")

def test_invalid_command():
    invalid_payload = {
        "command": "jump",  # Doesn't exist
        "dx": 1.0
    }
    print("\nTesting invalid command:")
    response = requests.post(BASE_URL + '/control', json=invalid_payload)
    print(f"Status: {response.status_code}")
    print(f"Response: {response.text}")

def test_malformed_json():
    malformed_data = "{not_valid_json}"
    print("\nTesting malformed JSON:")
    response = requests.post(
        BASE_URL + '/control',
        data=malformed_data,
        headers={"Content-Type": "application/json"}
    )
    print(f"Status: {response.status_code}")
    print(f"Response: {response.text}")

def test_stand(up: bool = True):
    stand_payload = {
        "command": "stand_up" if up else "stand_down"
    }
    print("\nTesting 'stand' command:")
    response = requests.post(BASE_URL + '/control', json=stand_payload)
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")

def test_get_state():
    print("\nTesting 'get_state' command:")
    response = requests.get(BASE_URL + '/state')
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")

def test_control_api():
    print("Testing Control API...")
    test_move(1.0, 0.0)
    test_move(-1.0, 0.0)
    test_rotate(90)
    test_rotate(-90)
    test_invalid_command()
    test_malformed_json()
    test_stand(False)
    test_stand(True)

if __name__ == "__main__":
    # test_control_api()
    gstreamer_test('./cache')
    # test_get_state()