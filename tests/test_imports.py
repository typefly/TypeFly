print("Testing imports...")

try:
    import numpy as np
    print(f"✅ numpy: {np.__version__}")
except Exception as e:
    print(f"❌ numpy: {e}")

try:
    import torch
    print(f"✅ torch: {torch.__version__}")
except Exception as e:
    print(f"❌ torch: {e}")

try:
    import cv2
    r = cv2.face.LBPHFaceRecognizer_create()
    print(f"✅ cv2: {cv2.__version__} + LBPH OK")
except Exception as e:
    print(f"❌ cv2: {e}")

try:
    from overrides import overrides
    print("✅ overrides: OK")
except Exception as e:
    print(f"❌ overrides: {e}")

try:
    from filterpy.kalman import KalmanFilter
    print("✅ filterpy: OK")
except Exception as e:
    print(f"❌ filterpy: {e}")

try:
    from facenet_pytorch import MTCNN, InceptionResnetV1
    print("✅ facenet_pytorch: OK")
except Exception as e:
    print(f"❌ facenet_pytorch: {e}")

try:
    import openai
    print(f"✅ openai: OK")
except Exception as e:
    print(f"❌ openai: {e}")

try:
    import quart
    print(f"✅ quart: OK")
except Exception as e:
    print(f"❌ quart: {e}")

print("\nDone!")