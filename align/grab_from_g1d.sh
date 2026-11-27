#!/bin/bash
# Read-only: copy g1d_grab_images.py to the G1-D, subscribe to its image streams for a few seconds and bring the
# frames back here. Never touches the arms, DDS, the image service or the camera settings.
# Needs the reverse tunnel from the PC (127.0.0.1:12222 on this login machine -> G1-D ssh), see SKILL.md.
#   grab_from_g1d.sh [seconds=10] [fps=5] [dest=logs/color_check/live_<stamp>]
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
SEC=${1:-10}; FPS=${2:-5}
DEST=${3:-$HERE/live_$(date +%Y%m%d_%H%M%S)}
SSH="ssh -p 12222 -o BatchMode=yes -o ConnectTimeout=8 -i $HOME/.ssh/id_g1d unitree@127.0.0.1"
SCP="scp -q -P 12222 -o BatchMode=yes -i $HOME/.ssh/id_g1d"
$SSH true || { echo "no ssh to the G1-D on 127.0.0.1:12222: start the tunnel on the PC first"; exit 1; }
$SCP "$HERE/g1d_grab_images.py" unitree@127.0.0.1:/tmp/g1d_grab_images.py
REMOTE=$($SSH "/home/unitree/miniconda3/envs/tv/bin/python /tmp/g1d_grab_images.py --seconds $SEC --fps $FPS" | tee /dev/stderr | sed -n 's/^saved //p')
[ -n "$REMOTE" ] || { echo "grab failed"; exit 1; }
mkdir -p "$(dirname "$DEST")"
$SCP -r "unitree@127.0.0.1:$REMOTE" "$DEST"
echo "frames in $DEST (head_left_* = color_0, the sim cam_head view)"
