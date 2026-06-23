#!/usr/bin/env bash
set -euo pipefail

# Some OpenCV/FFmpeg/decord stacks on Linux need libffi preloaded. Override or
# clear HAWOR_CLIP_LD_PRELOAD from the caller if this path is different.
export LD_PRELOAD="${HAWOR_CLIP_LD_PRELOAD:-/usr/lib/x86_64-linux-gnu/libffi.so.7}${LD_PRELOAD:+:$LD_PRELOAD}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python "$SCRIPT_DIR/heuristic_video_clipper.py" "$@"
