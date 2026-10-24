#!/bin/sh
# Install the segmentation runtime onto the Arduino UNO Q MPU.
# Run from this directory, on the board:  sh install.sh
set -eu
cd "$(dirname "$0")"

python3 - << 'PY'
import os
import sys

stats = os.statvfs(".")
free_mb = stats.f_bavail * stats.f_frsize // (1024 * 1024)
print(f"Free space: {free_mb} MiB")
if free_mb < 1024:
    sys.exit(
        "Need at least 1024 MiB free. The /home/arduino partition is often "
        "about 3.6 GB. Put this app under /opt, or run `arduino-app-cli system cleanup`."
    )
if sys.version_info < (3, 11):
    sys.exit("Python 3.11 or newer is required. The UNO Q image ships 3.13.")
PY

if command -v apt-get >/dev/null 2>&1; then
  sudo apt-get update
  sudo apt-get install -y libegl1 libgles2
fi

python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install --no-deps "mediapipe==1.1.1"
python -m face_shirt --self-check
python -m face_shirt --fetch
