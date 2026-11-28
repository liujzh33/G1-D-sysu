"""A copy of the school lab room (Assets/Room/real_scene4) under a new name with its own lights: the four ceiling panels
and the ambient dome of real_scene4/usd/lights.usda with another intensity, colour temperature and tint. Layouts pick
the room by name (Room.default), so real_scene4 and the data made in it stay as they are.

Usage (repo root, any python):
    python scripts/g1d/school_lights.py --name real_scene4_v3 --panel 18000 --temperature 5500 --dome 120 \
        --dome-color 1 1 1 [--tint 1 1 1]
"""

import argparse
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[2]
SOURCE = "real_scene4"
PANELS = ((-1, -1), (1, -1), (-1, 1), (1, 1))  # m, as real_scene4
PANEL_Z, PANEL_SIZE = 2.78, (1.2, 0.6)


def lights_usda(panel, temperature, tint, dome, dome_color):
    rects = "".join(
        f"""
    def RectLight "CeilingPanel_{i}"
    {{
        color3f inputs:color = ({tint[0]:g}, {tint[1]:g}, {tint[2]:g})
        float inputs:colorTemperature = {temperature:g}
        bool inputs:enableColorTemperature = 1
        float inputs:height = {PANEL_SIZE[1]:g}
        float inputs:intensity = {panel:g}
        float inputs:width = {PANEL_SIZE[0]:g}
        double3 xformOp:translate = ({x}, {y}, {PANEL_Z:g})
        uniform token[] xformOpOrder = ["xformOp:translate"]
    }}
"""
        for i, (x, y) in enumerate(PANELS)
    )
    return f"""#usda 1.0
(
    defaultPrim = "Lights"
    kilogramsPerUnit = 1
    metersPerUnit = 1
    upAxis = "Z"
)

def Xform "Lights"
{{{rects}
    def DomeLight "AmbientDome"
    {{
        color3f inputs:color = ({dome_color[0]:g}, {dome_color[1]:g}, {dome_color[2]:g})
        float inputs:intensity = {dome:g}
    }}
}}
"""


def write_room(name, panel, temperature, dome, dome_color=(1, 1, 1), tint=(1, 1, 1), assets=ROOT / "Assets"):
    src, dst = assets / "Room" / SOURCE, assets / "Room" / name
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst, symlinks=True)
    (dst / "usd" / "lights.usda").write_text(lights_usda(panel, temperature, tint, dome, dome_color))
    return dst


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True)
    ap.add_argument("--panel", type=float, default=24000, help="intensity of each ceiling panel (real_scene4: 24000)")
    ap.add_argument("--temperature", type=float, default=6500, help="K (real_scene4: 6500)")
    ap.add_argument("--tint", type=float, nargs=3, default=(1, 1, 1), help="panel colour on top of the temperature")
    ap.add_argument("--dome", type=float, default=150, help="ambient dome intensity (real_scene4: 150)")
    ap.add_argument("--dome-color", type=float, nargs=3, default=(0.95, 0.95, 1), help="(real_scene4: 0.95 0.95 1)")
    args = ap.parse_args()
    if args.name == SOURCE:
        raise SystemExit(f"{SOURCE} is the source room: pick another name")
    print(write_room(args.name, args.panel, args.temperature, args.dome, args.dome_color, args.tint))


if __name__ == "__main__":
    main()
