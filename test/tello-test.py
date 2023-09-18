from djitellopy import Tello
from threading import Thread
import cv2, time

class TelloLLM():
    def __init__(self):
        self.drone = Tello()
        self.drone.connect()
        self.battery = self.drone.query_battery()
        
    def check_battery(self):
        self.battery = self.drone.query_battery()
        print(f"> Battery level: {self.battery}% ", end='')
        if self.battery < 30:
            print('is too low [WARNING]')
        else:
            print('[OK]')
            return True
        return False

    def start(self):
        if not self.check_battery():
            return
        
        self.drone.streamon()
        self.streamOn = True
        self.drone.takeoff()
        print("> Application Start")

        self.drone.rotate_counter_clockwise(30)

        # aliveCount = 1
        # while (True):
        #     aliveCount += 1
        #     if aliveCount % 50 == 0:
        #         self.check_battery()
        #     frame = self.drone.get_frame_read().frame
        #     print("### GET Frame: ", frame.shape)
        #     cv2.imshow("Tello", frame)
        #     key = cv2.waitKey(10) & 0xff
        #     # Press esc to exit
        #     if key == 27:
        #         break

        self.drone.land()
        self.drone.streamoff()

def main():
    tello = TelloLLM()
    tello.start()

if __name__ == "__main__":
    main()