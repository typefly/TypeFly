import cv2
import numpy as np

def main():
    # Create a VideoCapture object
    cap = cv2.VideoCapture(1)

    camera_matrix = np.array([[336.58743799, 0, 324.59565213], [0, 339.37433598, 257.48248038], [0, 0, 1]])
    distortion_coeffs = np.array([5.63659207e-01, -1.47846360e+00, -1.21659229e-03,  6.56542001e-04, 8.67351081e-01])

    # Check if camera opened successfully
    if (cap.isOpened() == False):
        print("Unable to read camera feed")

    while True:
        ret, frame = cap.read()
        if ret == True:
            # Display the resulting frame
            frame = cv2.undistort(frame, camera_matrix, distortion_coeffs)
            cv2.imshow('Frame', frame)
            
            # Press Q on keyboard to exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break


if __name__ == '__main__':
    main()