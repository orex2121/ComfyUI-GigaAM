import importlib.metadata as meta
import os
import subprocess
import sys
import tempfile

GIGAAM_GIT = "gigaam @ git+https://github.com/salute-developers/GigaAM.git"


def is_installed(pkg: str) -> bool:
    try:
        meta.version(pkg)
        return True
    except meta.PackageNotFoundError:
        return False


# Preserve existing PyTorch stack AND any pre-existing torchcodec on ALL hardware
pins = [
    f"{pkg}=={meta.version(pkg)}"
    for pkg in ("torch", "torchaudio", "torchvision", "torchcodec")
    if is_installed(pkg)
]
had_torchcodec = is_installed("torchcodec")

with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
    f.write("\n".join(pins))
    lock_file = f.name

try:
    # GigaAM install + import check (aborts if broken)
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "--no-deps", GIGAAM_GIT]
    )
    subprocess.check_call([sys.executable, "-c", "import gigaam"])

    # Optional pyannote.audio install
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-c", lock_file, "pyannote.audio"]
        )
        # If pyannote freshly pulled in torchcodec, verify its C++/FFmpeg/GPU libs actually load
        if not had_torchcodec and is_installed("torchcodec"):
            probe = subprocess.call(
                [sys.executable, "-c", "import torchcodec.decoders"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            if probe != 0:
                print(
                    "[ComfyUI-GigaAM] Newly installed torchcodec failed to load; removing broken wheel..."
                )
                subprocess.call(
                    [sys.executable, "-m", "pip", "uninstall", "-y", "torchcodec"]
                )
    except subprocess.CalledProcessError:
        print(
            "[ComfyUI-GigaAM] WARNING: pyannote.audio failed to install. "
            "Longform ASR and speaker diarization will be unavailable."
        )
finally:
    try:
        os.remove(lock_file)
    except OSError:
        pass
