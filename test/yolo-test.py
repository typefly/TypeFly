import cv2
import torch
import numpy as np

def main():
    # Create a VideoCapture object
    cap = cv2.VideoCapture(2)
    camera_matrix = np.array([[336.58743799, 0, 324.59565213], [0, 339.37433598, 257.48248038], [0, 0, 1]])
    distortion_coeffs = np.array([5.63659207e-01, -1.47846360e+00, -1.21659229e-03,  6.56542001e-04, 8.67351081e-01])
    # load yolov5s model
    model = torch.hub.load("ultralytics/yolov5", "yolov5s")

    # Check if camera opened successfully
    if (cap.isOpened() == False):
        print("Unable to read camera feed")

    while True:
        ret, frame = cap.read()
        if ret == True:
            frame = cv2.undistort(frame, camera_matrix, distortion_coeffs)
            # Detect objects
            results = model(frame)
            # row: xmin, ymin, xmax, ymax, confidence, class, name
            print(results.pandas().xyxy[0])
            for index, row in results.pandas().xyxy[0].iterrows():
                if row['confidence'] < 0.1:
                    continue
                frame = cv2.putText(frame, row['name'], (int(row['xmin']), int(row['ymin']) - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 1, cv2.LINE_AA)
                frame = cv2.rectangle(frame, (int(row['xmin']), int(row['ymin'])), (int(row['xmax']), int(row['ymax'])), (0, 255, 0), 2)

            # Display the resulting frame
            cv2.imshow('Frame', frame)
            
            # Press Q on keyboard to exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break


if __name__ == '__main__':
    main()