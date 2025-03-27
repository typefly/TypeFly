# Go2 Setup

## Gstreamer
Create a `gstream-forward.sh` script with the following code to create a gstreamer stream on `eth0` then forward it to `wlan0`. You can run it directly or add it to systemmd service for auto-run after dog is up.
```
#!/bin/bash
gst-launch-1.0 -v \
  udpsrc address=230.1.1.1 port=1720 multicast-iface=eth0 \
  ! application/x-rtp, media=video, encoding-name=H264 \
  ! queue \
  ! udpsink host=230.1.1.1 port=1720 auto-multicast=true multicast-iface=wlan0
```

### Add the script to systemmd for auto-start
```
chmod +x gstream-forward.sh
sudo nano /etc/systemd/system/mystartup.service
```
Add this to `mystartup.service`, (replace the `/root/scripts/gstream-forward.sh` with your path of script):
```
[Unit]
Description=Run gstreamer forwarding
After=network.target

[Service]
ExecStart=/root/scripts/gstream-forward.sh
Restart=on-failure
User=root

[Install]
WantedBy=multi-user.target
```

Enable the service:
```
sudo systemctl daemon-reexec
sudo systemctl daemon-reload
sudo systemctl enable mystartup.service
```

Start it manully for test:
```
sudo systemctl start mystartup.service
```

### Server side
Install the gstreamer and python packages.
```
sudo apt install gstreamer1.0-plugins-good gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly gstreamer1.0-libav python3-gst-1.0 python3-gi
```