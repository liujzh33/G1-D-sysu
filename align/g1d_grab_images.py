"""Read-only image grab from a G1-D: subscribes to the teleimager ZMQ streams (ImageClient), never touches the arms,
DDS, the image service or the camera settings. Run ON the robot (tv conda env):

    /home/unitree/miniconda3/envs/tv/bin/python /tmp/g1d_grab_images.py [--seconds 10] [--fps 5] [--out /tmp/g1d_grab]

Writes <out>/<stamp>/: head_left_NNN.jpg (color_0), head_right_NNN.jpg (color_1), left_wrist_NNN.jpg, right_wrist_NNN.jpg
(the raw server JPEGs, decoded + re-encoded at q95 only to split the stereo pair), stats.json (per frame mean RGB and
luminance per camera, for auto exposure / white balance drift). ImageClient also refreshes its local cache
cam_config_client.yaml; that is a client-side file only.
"""
import argparse
import json
import os
import sys
import time

import cv2
import numpy as np

REPO = "/home/unitree/G1D-infra-main"
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "teleop/teleimager/src"))
from teleimager.image_client import ImageClient  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--seconds", type=float, default=10.0)
ap.add_argument("--fps", type=float, default=5.0)
ap.add_argument("--out", default="/tmp/g1d_grab")
ap.add_argument("--host", default="192.168.123.164")
args = ap.parse_args()
out = os.path.join(args.out, time.strftime("%Y%m%d_%H%M%S"))
os.makedirs(out, exist_ok=True)
c = ImageClient(host=args.host, request_bgr=True)
getters = dict(head=c.get_head_frame, left_wrist=c.get_left_wrist_frame, right_wrist=c.get_right_wrist_frame)
stats = {k: [] for k in ("head_left", "head_right", "left_wrist", "right_wrist")}
t0, n = time.time(), 0
while time.time() - t0 < args.seconds:
    tick = time.time()
    for name, get in getters.items():
        try:
            f = get()
        except Exception:
            f = None
        if f is None or getattr(f, "bgr", None) is None:
            continue
        img = f.bgr
        views = {"head_left": img[:, : img.shape[1] // 2], "head_right": img[:, img.shape[1] // 2:]} if name == "head" else {name: img}
        for v, im in views.items():
            cv2.imwrite(os.path.join(out, f"{v}_{n:03d}.jpg"), im, [cv2.IMWRITE_JPEG_QUALITY, 95])
            m = im.reshape(-1, 3).mean(0)[::-1]
            stats[v].append(dict(t=round(tick - t0, 3), rgb=m.round(2).tolist(), luma=round(float(m @ [0.299, 0.587, 0.114]), 2)))
    n += 1
    time.sleep(max(0.0, 1 / args.fps - (time.time() - tick)))
summary = {}
for v, s in stats.items():
    if s:
        L = np.array([x["luma"] for x in s])
        R = np.array([x["rgb"] for x in s])
        summary[v] = dict(frames=len(s), luma_mean=round(float(L.mean()), 2), luma_range=round(float(L.max() - L.min()), 2),
                          rgb_mean=R.mean(0).round(2).tolist(), rgb_range=(R.max(0) - R.min(0)).round(2).tolist())
json.dump(dict(summary=summary, frames=stats), open(os.path.join(out, "stats.json"), "w"), indent=1)
print(json.dumps(summary, indent=1))
print("saved", out)
