import cv2
from ultralytics import YOLO

# Open the default camera (0)
cap = cv2.VideoCapture(0)
model = YOLO("yolov8x.pt")
model.to("mps")

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

while True:
    ret, frame = cap.read()  # Capture frame-by-frame
    if not ret:
        print("Error: Could not read frame.")
        break

    # Perform object detection
    results = model(frame)


    cv2.imshow('Camera Feed', results[0].plot())  # Display the frame

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()  # Release the camera
cv2.destroyAllWindows()  # Close all OpenCV windows