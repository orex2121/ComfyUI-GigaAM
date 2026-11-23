import subprocess
import sys

GIGAAM_GIT = "gigaam @ git+https://github.com/salute-developers/GigaAM.git"


def install():
    print(
        "[ComfyUI-GigaAM] Installing GigaAM with --no-deps to preserve existing PyTorch build..."
    )
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "--no-deps", GIGAAM_GIT]
    )
    # Check GigaAM import
    subprocess.check_call(
        [
            sys.executable,
            "-c",
            "import gigaam; print('[ComfyUI-GigaAM] GigaAM OK:', gigaam.__file__)",
        ]
    )


if __name__ == "__main__":
    install()
