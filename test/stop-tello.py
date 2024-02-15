from djitellopy import Tello

tello = Tello()
tello.connect()
tello.streamoff()
tello.land()