"""Appearance variants of the tape-measured school table room (Assets/Room/real_scene4_v2) for datagen: copies named
<prefix>_<k>, each with its own ceiling lights, tabletop texture and green-screen colour. Layouts pick a variant by
name (Room.default); the base room stays as it is.

The base room's lights are the brightest a variant gets: every panel and the dome are scaled down (overall factor
log-uniform in LIGHT_SCALE, each panel a little more on its own for the light's direction), at the base's power or
below whatever the panel size. Colours are centred on the real head camera (data/real/g1-d-sysu, episodes 34 / 36,
against the first school_v2 collection, linear sRGB of the masked table / cloth pixels): with the light at 0.71 of the
base (the tabletop's green channel) the real tabletop is TABLE_TINT warmer and the real cloth CLOTH_TARGET of the
base cloth, i.e. much less yellow and darker. Each draw mixes from the base towards that centre (a little past it).
The tabletop map (usd/textures/board.png) is tinted, scaled and, in GRAIN_SHARE of the variants, overlaid with the
grain of another light wood (school_variants.grain_map).

Usage (repo root, python with numpy + PIL):
    python scripts/g1d/school_v2_variants.py --prefix school_v2v --count 64 --seed 0
"""

import argparse
import json
from pathlib import Path
import re
import shutil
import sys

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.g1d.school_variants import WOODS_V4A, grain_map  # noqa: E402

SOURCE = "real_scene4_v2"
LIGHT_SCALE = (0.3, 1.0)  # overall light, x the base room's (log-uniform)
PANEL_SPREAD = (0.6, 1.0)  # each panel on top of that
DOME_SPREAD = (0.5, 1.0)
TEMPERATURE = (4000, 6500)  # K (base 6000)
PANEL_SCALE = (0.7, 1.4)  # panel size, at constant power: smaller = harder shadows
PANEL_SHIFT = 0.35  # m, the panel grid moved in x / y: shadow direction
TABLE_TINT = np.array([1.14, 1.0, 0.89])  # real / base tabletop per linear channel, normalised to green
TABLE_MIX, TABLE_GAIN, TABLE_ROUGHNESS = (0.0, 1.3), (0.85, 1.0), (0.35, 0.7)
CLOTH_TARGET = np.array([0.20, 0.56, 1.03])  # real / base cloth per linear channel, at the same light
CLOTH_MIX, CLOTH_GAIN = (0.4, 1.15), (0.85, 1.05)
GRAIN_SHARE, GRAIN_STRENGTH, GRAIN_TILE = 0.6, (0.15, 0.5), (0.4, 1.0)  # tile: board fraction per repeat


def srgb_to_lin(x):
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def lin_to_srgb(x):
    x = np.clip(x, 0, 1)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * x ** (1 / 2.4) - 0.055)


def draw(rng):
    u = lambda r: float(rng.uniform(*r))  # noqa: E731
    s = float(np.exp(rng.uniform(*np.log(LIGHT_SCALE))))
    cloth_mix = u(CLOTH_MIX)
    v = dict(
        light_scale=s,
        panels=[s * u(PANEL_SPREAD) for _ in range(4)],
        dome=s * u(DOME_SPREAD),
        temperature=u(TEMPERATURE),
        panel_scale=u(PANEL_SCALE),
        panel_shift=[float(x) for x in rng.uniform(-PANEL_SHIFT, PANEL_SHIFT, 2)],
        table_tint=(1 + u(TABLE_MIX) * (TABLE_TINT - 1)).tolist(),
        table_gain=u(TABLE_GAIN),
        table_roughness=u(TABLE_ROUGHNESS),
        cloth_factor=((1 + cloth_mix * (CLOTH_TARGET - 1)) * u(CLOTH_GAIN)).tolist(),
    )
    if rng.random() < GRAIN_SHARE:
        key = str(rng.choice(sorted(WOODS_V4A)))
        v["grain"] = dict(map=f"{key}/{WOODS_V4A[key]}", strength=u(GRAIN_STRENGTH), tile=u(GRAIN_TILE),
                          rotation=int(rng.choice([0, 90])), offset=[float(x) for x in rng.random(2)])  # fmt: skip
    return v


def lights_usda(base, v):
    """The base lights.usda with each panel's intensity, size, colour temperature and position and the dome's
    intensity replaced."""
    out = base
    for i, f in enumerate(v["panels"]):
        m = re.search(rf'def RectLight "CeilingPanel_{i}"\s*\{{.*?\n    \}}', out, re.S)
        b = m.group(0)
        w, h = (float(re.search(rf"float inputs:{k} = ([0-9.]+)", b).group(1)) for k in ("width", "height"))
        I = float(re.search(r"float inputs:intensity = ([0-9.]+)", b).group(1))
        x, y, z = (float(c) for c in re.search(r"xformOp:translate = \(([^)]*)\)", b).group(1).split(","))
        k = v["panel_scale"]
        b = re.sub(r"float inputs:intensity = [0-9.]+", f"float inputs:intensity = {I * f / k**2:.0f}", b)
        b = re.sub(r"float inputs:width = [0-9.]+", f"float inputs:width = {w * k:.3f}", b)
        b = re.sub(r"float inputs:height = [0-9.]+", f"float inputs:height = {h * k:.3f}", b)
        b = re.sub(r"float inputs:colorTemperature = [0-9.]+", f"float inputs:colorTemperature = {v['temperature']:.0f}", b)
        sx, sy = v["panel_shift"]
        b = re.sub(r"xformOp:translate = \([^)]*\)", f"xformOp:translate = ({x + sx:.3f}, {y + sy:.3f}, {z:g})", b)
        out = out[: m.start()] + b + out[m.end() :]
    m = re.search(r'def DomeLight "AmbientDome"\s*\{.*?\n    \}', out, re.S)
    b = m.group(0)
    I = float(re.search(r"float inputs:intensity = ([0-9.]+)", b).group(1))
    b = re.sub(r"float inputs:intensity = [0-9.]+", f"float inputs:intensity = {I * v['dome']:.1f}", b)
    return out[: m.start()] + b + out[m.end() :]


def board_texture(src, v, assets):
    """The tabletop map tinted and scaled in linear light, with another wood's grain multiplied in."""
    x = srgb_to_lin(np.asarray(Image.open(src).convert("RGB"), np.float64) / 255)
    x = x * np.asarray(v["table_tint"]) * v["table_gain"]
    if "grain" in v:
        g = v["grain"]
        gm = np.asarray(Image.open(assets / "Material" / grain_map(assets, g["map"])), np.float64) / 255  # mean 0.5
        if g["rotation"]:
            gm = np.rot90(gm)
        h, w = x.shape[:2]
        n = int(round(max(h, w) * g["tile"]))
        tile = np.asarray(Image.fromarray((gm * 255).astype(np.uint8)).resize((n, n), Image.BILINEAR), np.float64) / 255
        oy, ox = (int(o * n) for o in g["offset"])
        reps = (int(np.ceil((h + oy) / n)) + 1, int(np.ceil((w + ox) / n)) + 1)
        tile = np.tile(tile, reps)[oy : oy + h, ox : ox + w]
        x = x * (1 + g["strength"] * (tile / 0.5 - 1))[..., None]
    return Image.fromarray((lin_to_srgb(x) * 255).round().astype(np.uint8))


def set_diffuse(text, name, factor):
    m = re.search(rf'def Material "{name}"\s*\{{.*?\n        \}}', text, re.S)
    b = m.group(0)
    c = np.array([float(t) for t in re.search(r"color3f inputs:diffuseColor = \(([^)]*)\)", b).group(1).split(",")])
    c = ", ".join(f"{t:.4f}" for t in np.clip(c * factor, 0, 1))
    b = re.sub(r"color3f inputs:diffuseColor = \([^)]*\)", f"color3f inputs:diffuseColor = ({c})", b)
    return text[: m.start()] + b + text[m.end() :]


def set_roughness(text, name, roughness):
    m = re.search(rf'def Material "{name}"\s*\{{.*?\n        \}}', text, re.S)
    b = re.sub(r"float inputs:roughness = [0-9.]+", f"float inputs:roughness = {roughness:.2f}", m.group(0))
    return text[: m.start()] + b + text[m.end() :]


def write_variant(name, v, assets=ROOT / "Assets"):
    src, dst = assets / "Room" / SOURCE, assets / "Room" / name
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst, symlinks=True)
    usd = dst / "usd"
    (usd / "lights.usda").write_text(lights_usda((src / "usd" / "lights.usda").read_text(), v))
    board_texture(src / "usd" / "textures" / "board.png", v, assets).save(usd / "textures" / "board.png")
    (usd / "table.usda").write_text(set_roughness((usd / "table.usda").read_text(), "board", v["table_roughness"]))
    (usd / "background.usda").write_text(set_diffuse((usd / "background.usda").read_text(), "screen_green", v["cloth_factor"]))
    (dst / "variant.json").write_text(json.dumps(dict(v, source=SOURCE), indent=2) + "\n")
    return dst


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prefix", default="school_v2v")
    ap.add_argument("--count", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    for k in range(args.count):
        v = draw(rng)
        write_variant(f"{args.prefix}_{k}", v)
        print(f"{args.prefix}_{k}: light x{v['light_scale']:.2f} {v['temperature']:.0f} K panel x{v['panel_scale']:.2f} "
              f"table tint {np.round(v['table_tint'], 2).tolist()} x{v['table_gain']:.2f} "
              f"cloth x{np.round(v['cloth_factor'], 2).tolist()} {v.get('grain', {}).get('map', '-')}")  # fmt: skip


if __name__ == "__main__":
    main()
