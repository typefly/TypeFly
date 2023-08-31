from djitellopy import Tello
import cv2, time, os
import openai
from TelloWrapper import TelloWrapper

USE_DRONE = False

openai.organization = "org-sAnQwPNnbSrHg1XyR4QYALf7"
openai.api_key = os.environ.get('OPENAI_API_KEY')

class TelloLLM():
    def __init__(self):
        self.model_id = "gpt-3.5-turbo-16k"
        if USE_DRONE:
            self.tello = TelloWrapper()
            self.tello.start()
        else:
            self.cap = cv2.VideoCapture(1)
            # Check if camera opened successfully
            if (self.cap.isOpened() == False):
                print("Unable to read camera feed")

    def get_image(self):
        if USE_DRONE:
            return self.tello.get_image()
        else:
            ret, frame = self.cap.read()
            if ret == True:
                return frame
            return None

def main():
    tellollm = TelloLLM()
    while True:
        frame = tellollm.get_image()
        # Display the resulting frame
        cv2.imshow('Frame', frame)
        
        # Press Q on keyboard to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

if __name__ == "__main__":
    main()