import sys
sys.path.append("/usr/lib/python3/dist-packages")
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib

import cv2
import numpy as np

# Initialize GStreamer
Gst.init(None)

# Define the pipeline with appsink
pipeline_str = """
udpsrc address=230.1.1.1 port=1720 multicast-iface=wlan0
! application/x-rtp, media=video, encoding-name=H264
! rtph264depay
! h264parse
! avdec_h264
! videoconvert
! video/x-raw, format=BGR
! appsink name=appsink emit-signals=true max-buffers=1 drop=true
"""

# Create the pipeline
pipeline = Gst.parse_launch(pipeline_str)

# Get the appsink element
appsink = pipeline.get_by_name("appsink")
appsink.set_property("emit-signals", True)
appsink.set_property("sync", False)

# Callback to convert buffer to OpenCV image
def on_new_sample(sink):
    sample = sink.emit("pull-sample")
    if sample:
        buffer = sample.get_buffer()
        caps = sample.get_caps()
        width = caps.get_structure(0).get_value('width')
        height = caps.get_structure(0).get_value('height')

        success, map_info = buffer.map(Gst.MapFlags.READ)
        if success:
            # Convert to NumPy array
            frame = np.frombuffer(map_info.data, np.uint8)
            frame = frame.reshape((height, width, 3))  # BGR format
            buffer.unmap(map_info)

            print(f"Frame size: {frame.shape}")

            # Show the frame using OpenCV
            # cv2.imshow("Video", frame)
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     loop.quit()
        return Gst.FlowReturn.OK
    return Gst.FlowReturn.ERROR

# Connect callback
appsink.connect("new-sample", on_new_sample)

# Start the pipeline
pipeline.set_state(Gst.State.PLAYING)

# Set up GLib MainLoop
loop = GLib.MainLoop()

try:
    print("Starting main loop... Press 'q' to quit.")
    loop.run()
except KeyboardInterrupt:
    print("Interrupted by user")
finally:
    # Clean up
    pipeline.set_state(Gst.State.NULL)
    cv2.destroyAllWindows()