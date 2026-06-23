"""Ensure the gRPC stubs exist before they are imported.

The generated stubs (``hyrch_serving_pb2.py`` / ``hyrch_serving_pb2_grpc.py``)
are build artifacts derived from the tracked ``hyrch_serving.proto`` and are
gitignored, so a fresh clone does not contain them. ``ensure_stubs()`` generates
them on demand the first time the serving code is imported, removing the manual
``bash generate.sh`` step. ``grpc_tools`` is already a project dependency.
"""

import os
import subprocess
import sys

_PROTO_DIR = os.path.dirname(os.path.abspath(__file__))
_STUBS = ("hyrch_serving_pb2.py", "hyrch_serving_pb2_grpc.py")


def ensure_stubs() -> None:
    """Generate the gRPC stubs from the .proto if they are missing (idempotent)."""
    if all(os.path.exists(os.path.join(_PROTO_DIR, name)) for name in _STUBS):
        return

    print("[typefly] Generating gRPC stubs from hyrch_serving.proto ...")
    try:
        subprocess.run(
            [sys.executable, "-m", "grpc_tools.protoc",
             "-I.", "--python_out=.", "--grpc_python_out=.", "hyrch_serving.proto"],
            cwd=_PROTO_DIR,
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        raise RuntimeError(
            f"Failed to auto-generate gRPC stubs ({e}). "
            f"Generate them manually with: cd {_PROTO_DIR} && bash generate.sh"
        ) from e
