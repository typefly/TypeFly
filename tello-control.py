from djitellopy import Tello
from threading import Thread
import cv2, time

tello = Tello()
tello.connect()

tello.streamon()
streamOn = True
frame_read = tello.get_frame_read()

def videoRecorder():
    height, width, _ = frame_read.frame.shape

    while streamOn:
        # name the image file with time stamp
        filename = f"images/{time.strftime('%Y%m%d-%H%M%S')}.jpg"
        cv2.imwrite(filename, frame_read.frame)
        time.sleep(1 / 30)

tello.takeoff()

tello.move_left(20)

tello.land()
streamOn = False