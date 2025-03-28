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
Install the [gstreamer](https://gstreamer.freedesktop.org/documentation/installing/on-linux.html?gi-language=c) and compile opencv `DWITH_GSTREAMER=ON`.
```
sudo apt install libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev libgstreamer-plugins-bad1.0-dev gstreamer1.0-plugins-base gstreamer1.0-plugins-good gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly gstreamer1.0-libav gstreamer1.0-tools gstreamer1.0-x gstreamer1.0-alsa gstreamer1.0-gl gstreamer1.0-gtk3 gstreamer1.0-qt5 gstreamer1.0-pulseaudio
```
Compile `opencv` (we recommand using a new conda env with numpy installed to compile it):
```
git clone --recursive https://github.com/skvark/opencv.git
git clone --recursive https://github.com/opencv/opencv_contrib.git
cd opencv
mkdir build && cd build
cmake -D CMAKE_BUILD_TYPE=Release \
      -D OPENCV_GENERATE_PKGCONFIG=ON \
      -D BUILD_EXAMPLES=OFF \
      -D BUILD_TESTS=OFF \
      -D BUILD_PERF_TESTS=OFF \
      -D BUILD_DOCS=OFF \
      -D BUILD_opencv_python2=OFF \
      -D BUILD_opencv_python3=ON \
      -D WITH_GSTREAMER=ON \
      -D WITH_GSTREAMER_0_10=OFF \
      -D PYTHON3_EXECUTABLE=$(which python) \
      -D PYTHON3_INCLUDE_DIR=$(python -c "from sysconfig import get_paths as p; print(p()['include'])") \
      -D PYTHON3_PACKAGES_PATH=$(python -c "from distutils.sysconfig import get_python_lib; print(get_python_lib())") \
      -D PYTHON3_NUMPY_INCLUDE_DIRS=$(python -c "import numpy; print(numpy.get_include())") \
      -D OPENCV_PYTHON3_INSTALL_PATH=$(python -c "from distutils.sysconfig import get_python_lib; print(get_python_lib())") \
      -D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules \
      ..
make -j$(nproc)
cd python_loader & python setup.py bdist_wheel
# install the package in any env you want to use
pip install dist/opencv*.whl
```