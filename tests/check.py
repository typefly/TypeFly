import sys
print("="*50)
print("PYTHON:", sys.version)
print("="*50)

packages = {
    'torch': 'torch',
    'cv2': 'cv2', 
    'ultralytics': 'ultralytics',
    'grpc': 'grpc',
    'flask': 'flask',
    'numpy': 'numpy',
    'PIL': 'PIL',
    'facenet_pytorch': 'facenet_pytorch',
    'djitellopy': 'djitellopy',
}

for name, pkg in packages.items():
    try:
        mod = __import__(pkg)
        ver = getattr(mod, '__version__', 'installed')
        print(f"OK  {name}: {ver}")
    except ImportError:
        print(f"MISSING  {name}")

print("="*50)
try:
    import torch
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("Running on CPU")
except Exception as e:
    print(f"torch error: {e}")
print("="*50)