from djitellopy import Tello

class TelloWrapper():
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

    def keep_alive(self):
        self.get_state()

    def start(self):
        if not self.check_battery():
            return
        self.drone.streamon()
        self.streamOn = True
        self.drone.takeoff()

    def get_image(self):
        if not self.streamOn:
            return None
        return self.drone.get_frame_read().frame

    def stop(self):
        self.drone.land()
        self.streamOn = False
        self.drone.streamoff()