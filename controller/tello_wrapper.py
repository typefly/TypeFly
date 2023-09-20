from djitellopy import Tello

class TelloWrapper():
    def __init__(self):
        self.drone = Tello()
        self.active_count = 0
        
    def check_battery(self):
        self.battery = self.drone.query_battery()
        print(f"> Battery level: {self.battery}% ", end='')
        if self.battery < 20:
            print('is too low [WARNING]')
        else:
            print('[OK]')
            return True
        return False
    
    def connect(self):
        self.drone.connect()

    def keep_alive(self):
        if self.active_count % 20 == 0:
            self.drone.send_control_command("command")
        self.active_count += 1

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
        self.drone.streamoff()
        self.streamOn = False
