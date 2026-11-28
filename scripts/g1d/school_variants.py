"""Appearance variants of the school lab room for datagen: copies of Assets/Room/real_scene4 named <prefix>_<k>, each
with its own ceiling lights (intensity, colour temperature, panel size and spot: brightness, shadow softness and
direction), ambient dome, tabletop colour and roughness, and green-screen colour. Layouts pick a variant by name
(Room.default), so the base room and the data made in it stay as they are.

Every draw stays around what the real robot's head camera sees (deploy frames): ceiling panels around the relit
real_scene4_v3 (13500, 5500 K, scripts/g1d/school_lights.py), the tabletop between the twin's yellow wood and the real
table's pinker wood, the screen cloth between the twin's yellow-green and the real cloth's bluer green, its frame
tubes mostly covered (real: not seen) and sometimes dark as in the twin.

--textured (school_v4, docs/superpowers/specs/2026-11-11-g1d-house4-v4-design.md): the same draws, and on top a wood
grain on the tabletop and a weave on the screen cloth (each in TEXTURED share of the variants) from Assets/Material's
base colour maps. Only a map's brightness pattern is used (grain_map: normalised to one contrast), on the drawn colour
with its contrast scaled by `strength`, so the look stays around the real table and cloth; tile size, rotation and
offset are baked into the UVs written onto the meshes (the room's meshes have none).

Usage (repo root, any python):
    python scripts/g1d/school_variants.py --prefix school_v3 --count 32 --seed 0
    python scripts/g1d/school_variants.py --prefix school_v4 --count 32 --seed 0 --textured
    python scripts/g1d/school_variants.py --prefix school_v4a --count 64 --seed 0 --v4a
"""

import argparse
import json
from pathlib import Path
import re
import shutil
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.g1d.school_lights import PANEL_SIZE, PANEL_Z, SOURCE  # noqa: E402

# UsdPreviewSurface diffuse colours (linear)
TABLE_TWIN, TABLE_REAL = np.array([0.4563, 0.3371, 0.1726]), np.array([0.53, 0.37, 0.31])
CLOTH_TWIN, CLOTH_REAL = np.array([0.0039, 0.1483, 0.0112]), np.array([0.036, 0.33, 0.16])
SLEEVE_TWIN = np.array([0.004212, 0.160164, 0.012096])
FRAME_TWIN = np.array([0.018, 0.022, 0.021])
RANGES = dict(
    panel=(9000, 15500),  # intensity of each ceiling panel
    temperature=(4800, 6300),  # K
    dome=(50, 130),
    panel_scale=(0.6, 1.5),  # panel size: smaller = harder shadows
    panel_shift=0.35,  # m, the panel grid moved in x / y: shadow direction
    table_mix=(0.3, 1.1),  # 0 twin wood, 1 real wood
    table_gain=(0.88, 1.12),
    table_roughness=(0.4, 0.75),
    cloth_mix=(0.5, 1.1),  # 0 twin green, 1 real green
    cloth_gain=(0.85, 1.15),
    frame_covered=0.8,  # share of variants whose frame tubes take the cloth colour
)
# --textured: base colour maps (Assets/Material/<id>/<file>): light woods without plank seams; plain grey / white weaves
WOODS = {
    "material_0111": "Ash_BaseColor.png", "material_0116": "Birch_BaseColor.png",
    "material_0118": "Cherry_BaseColor.png", "material_0123": "Oak_BaseColor.png",
    "material_0126": "Plywood_BaseColor.png", "material_0555": "laminate_oak_diffuse.jpg",
    "material_0282": "ov_wood_grain_d_01_basecolor.png", "material_0283": "ov_wood_grain_e_01_basecolor.png",
}  # fmt: skip
WEAVES = {  # not Cloth_Gray / Linen_White: their speckle reads as grass on the screen
    "material_0163": "ov_fabric_grain_01_basecolor.png", "material_0168": "ov_fabric_polyester_03_basecolor.png",
    "material_0170": "ov_fabric_weave_01_basecolor.png", "material_0171": "ov_fabric_weave_02_basecolor.png",
    "material_0322": "cotton_roughly_woven_diff.jpg", "material_0325": "fine_woven_cotton_diff.jpg",
}  # fmt: skip
TEXTURED = 0.8  # share of variants with a map, per surface
TEXTURE_RANGES = dict(
    wood=dict(tile=(0.5, 1.2), strength=(0.4, 1.0)),  # m per map repeat; contrast kept (0 = flat colour)
    weave=dict(tile=(0.25, 0.8), strength=(0.2, 0.6)),
)
# --v4a (school_v4a): colours centred on the real head camera, from runs/sim_real_gap (v3e rooms against 3 real
# stack_cube episodes, head view). Cloth: real sRGB (8, 101, 70) against sim (31, 129, 73), i.e. x(0.18, 0.59, 0.92)
# per linear channel on the v3 rooms' mean cloth (mix 0.8): darker, much less red, bluer green. Tabletop: real
# (199, 181, 157) against (189, 167, 141), x(1.12, 1.20, 1.27) on the mean v3 tabletop: a little lighter and whiter.
# The draw mixes from the v3 mean towards that centre (and a little past it), so the old look stays at the range's edge.
CLOTH_V3_MEAN, CLOTH_V4A = np.array([0.0296, 0.2937, 0.1302]), np.array([0.0052, 0.1742, 0.1197])
TABLE_V3_MEAN, TABLE_V4A = np.array([0.5079, 0.3601, 0.2688]), np.array([0.570, 0.431, 0.341])
RANGES_V4A = dict(RANGES, cloth_mix=(0.7, 1.15), cloth_gain=(0.85, 1.15), table_mix=(0.6, 1.2), table_gain=(0.92, 1.08),
                  frame_covered=0.95)  # fmt: skip
# the real cloth shows big folds, no weave (v4's weave put 2.9-3.7 of noise sigma on the cloth, real 1.9-2.1): a weak,
# coarse weave on fewer variants; no Oak (dark dense grain), a weaker grain on the tabletop
WOODS_V4A = {k: v for k, v in WOODS.items() if k != "material_0123"}
TEXTURED_V4A = dict(wood=0.8, weave=0.6)
TEXTURE_RANGES_V4A = dict(
    wood=dict(tile=(0.5, 1.2), strength=(0.25, 0.7)),
    weave=dict(tile=(0.6, 1.5), strength=(0.05, 0.25)),
)


def lights_usda(panel, temperature, dome, scale, shift):
    w, h = PANEL_SIZE[0] * scale, PANEL_SIZE[1] * scale
    rects = "".join(
        f"""
    def RectLight "CeilingPanel_{i}"
    {{
        float inputs:colorTemperature = {temperature:.0f}
        bool inputs:enableColorTemperature = 1
        float inputs:height = {h:.3f}
        float inputs:intensity = {panel / scale**2:.0f}
        float inputs:width = {w:.3f}
        double3 xformOp:translate = ({x + shift[0]:.3f}, {y + shift[1]:.3f}, {PANEL_Z:g})
        uniform token[] xformOpOrder = ["xformOp:translate"]
    }}
"""
        for i, (x, y) in enumerate(((-1, -1), (1, -1), (-1, 1), (1, 1)))
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
        color3f inputs:color = (1, 1, 1)
        float inputs:intensity = {dome:.0f}
    }}
}}
"""


def set_material(text, name, color=None, roughness=None):
    """Replace the diffuse colour / roughness inside `def Material "<name>"` of a Looks scope."""
    m = re.search(rf'def Material "{name}"\s*\{{.*?\n        \}}', text, re.S)
    block = m.group(0)
    if color is not None:
        c = ", ".join(f"{v:.4f}" for v in np.clip(color, 0, 1))
        block = re.sub(r"color3f inputs:diffuseColor = \([^)]*\)", f"color3f inputs:diffuseColor = ({c})", block)
    if roughness is not None:
        block = re.sub(r"float inputs:roughness = [0-9.]+", f"float inputs:roughness = {roughness:.2f}", block)
    return text[: m.start()] + block + text[m.end() :]


def draw(rng, v4a=False):
    R = RANGES_V4A if v4a else RANGES
    cloth_a, cloth_b = (CLOTH_V3_MEAN, CLOTH_V4A) if v4a else (CLOTH_TWIN, CLOTH_REAL)
    table_a, table_b = (TABLE_V3_MEAN, TABLE_V4A) if v4a else (TABLE_TWIN, TABLE_REAL)
    u = lambda key: float(rng.uniform(*R[key]))  # noqa: E731
    cloth = (cloth_a + u("cloth_mix") * (cloth_b - cloth_a)) * u("cloth_gain")
    return dict(
        panel=u("panel"),
        temperature=u("temperature"),
        dome=u("dome"),
        panel_scale=u("panel_scale"),
        panel_shift=[float(v) for v in rng.uniform(-R["panel_shift"], R["panel_shift"], 2)],
        table=((table_a + u("table_mix") * (table_b - table_a)) * u("table_gain")).tolist(),
        table_roughness=u("table_roughness"),
        cloth=cloth.tolist(),
        sleeve=(cloth * SLEEVE_TWIN[1] / CLOTH_TWIN[1]).tolist(),
        frame=(cloth * 0.8 if rng.random() < R["frame_covered"] else FRAME_TWIN).tolist(),
    )


def draw_textures(rng, v4a=False):
    out = {}
    for surface, maps in (("wood", WOODS_V4A if v4a else WOODS), ("weave", WEAVES)):
        if rng.random() >= (TEXTURED_V4A[surface] if v4a else TEXTURED):
            continue
        R = (TEXTURE_RANGES_V4A if v4a else TEXTURE_RANGES)[surface]
        key = str(rng.choice(sorted(maps)))
        # wood grain along the table's width or depth, a few degrees off; a weave at any angle
        turn = rng.uniform(0, 360) if surface == "weave" else rng.choice([0, 90]) + rng.uniform(-8, 8)
        out[surface] = dict(
            map=f"{key}/{maps[key]}",
            tile=float(rng.uniform(*R["tile"])),
            strength=float(rng.uniform(*R["strength"])),
            rotation_deg=float(turn),
            offset=[float(v) for v in rng.random(2)],
        )  # fmt: skip
    return out


GRAIN_DIR = "school_v4_maps"  # Assets/Material/<this>/<material id>.png
GRAIN_STD, GRAIN_SIZE = 0.25, 1024  # relative std of every grain map; pixels


def grain_map(assets, key):
    """Assets/Material/GRAIN_DIR/<id>.png of a base colour map: only its brightness pattern, linear, mean 0.5 and
    relative std GRAIN_STD (raw: value / 0.5 is the factor on the drawn colour), so every map tints to the drawn colour
    and the same strength gives the same contrast whatever the map's own colour and contrast."""
    from PIL import Image

    mid, name = key.split("/")
    dst = assets / "Material" / GRAIN_DIR / f"{mid}.png"
    if not dst.exists():
        x = Image.open(assets / "Material" / key).convert("RGB").resize((GRAIN_SIZE, GRAIN_SIZE))
        x = np.asarray(x, np.float64)
        x /= 255
        lin = np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)
        lum = lin @ [0.2126, 0.7152, 0.0722]
        rel = lum / lum.mean() - 1
        g = np.clip(0.5 * (1 + rel * GRAIN_STD / rel.std()), 0, 1)
        dst.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray((g * 255).round().astype(np.uint8)).save(dst)
    return f"{GRAIN_DIR}/{mid}.png"


def mesh_block(text, name, start=0):
    """(start, end) of the body of the next `def Mesh "<name>"` from `start`: up to its subdivisionScheme line."""
    m = re.compile(rf'def Mesh "{name}"').search(text, start)
    if m is None:
        return None
    end = text.index("uniform token subdivisionScheme", m.end())
    return m.start(), end


def parse_array(block, decl):
    m = re.search(rf"{re.escape(decl)} = \[(.*?)\]\n", block, re.S)
    return np.array([float(v) for v in re.findall(r"-?[0-9.]+(?:e-?[0-9]+)?", m.group(1))])


def planar_uv(p, axes, tex):
    """UVs of points p (N, 3) on the plane of `axes`, tile / rotation / offset from tex."""
    a = np.deg2rad(tex["rotation_deg"])
    rot = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    return (p[:, axes] @ rot.T) / tex["tile"] + tex["offset"]


def add_uvs(text, name, tex, face_varying):
    """primvars:st on every `def Mesh "<name>"`: per face corner on each face's own plane (face_varying: a box), else
    per point on the mesh's two widest axes (a sheet)."""
    pos = 0
    while (span := mesh_block(text, name, pos)) is not None:
        s, e = span
        block = text[s:e]
        p = parse_array(block, "point3f[] points").reshape(-1, 3)
        if face_varying:
            counts = parse_array(block, "int[] faceVertexCounts").astype(int)
            idx = parse_array(block, "int[] faceVertexIndices").astype(int)
            uv, k = [], 0
            for c in counts:
                q = p[idx[k : k + c]]
                axes = np.argsort(np.ptp(q, axis=0))[::-1][:2]
                uv.append(planar_uv(q, np.sort(axes), tex))
                k += c
            uv, interp = np.concatenate(uv), "faceVarying"
        else:
            uv, interp = planar_uv(p, np.sort(np.argsort(np.ptp(p, axis=0))[::-1][:2]), tex), "vertex"
        vals = ", ".join(f"({u:.4f}, {v:.4f})" for u, v in uv)
        ind = re.search(r"( *)$", text[:e]).group(1)  # the subdivisionScheme line's indent, already in place
        ins = f'texCoord2f[] primvars:st = [{vals}] (\n{ind}    interpolation = "{interp}"\n{ind})\n{ind}'
        text = text[:e] + ins + text[e:]
        pos = e + len(ins) + 1
    return text


def set_texture(text, name, path, tex, color, roughness, assets):
    """`def Material "<name>"` (prim path `path`) as the grain of a base colour map (grain_map) on `color`:
    out = grain * scale + bias with scale = 2 * strength * color, bias = (1 - strength) * color, so the mean stays
    `color` and the relative std is strength * GRAIN_STD."""
    m = re.search(rf'def Material "{name}"\s*\{{.*?\n        \}}', text, re.S)
    if roughness is None:
        roughness = float(re.search(r"float inputs:roughness = ([0-9.]+)", m.group(0)).group(1))
    color, k = np.clip(np.asarray(color, float), 0, 1), tex["strength"]
    grain = grain_map(assets, tex["map"])
    scale, bias = 2 * k * color, (1 - k) * color
    f4 = lambda v, w: ", ".join(f"{x:.4f}" for x in [*v, w])  # noqa: E731
    block = f'''def Material "{name}"
        {{
            token outputs:surface.connect = <{path}/Shader.outputs:surface>

            def Shader "Shader"
            {{
                uniform token info:id = "UsdPreviewSurface"
                color3f inputs:diffuseColor.connect = <{path}/Map.outputs:rgb>
                float inputs:roughness = {roughness:.2f}
                token outputs:surface
            }}

            def Shader "Map"
            {{
                uniform token info:id = "UsdUVTexture"
                asset inputs:file = @../../../Material/{grain}@
                token inputs:sourceColorSpace = "raw"
                float2 inputs:st.connect = <{path}/Uv.outputs:result>
                token inputs:wrapS = "repeat"
                token inputs:wrapT = "repeat"
                float4 inputs:scale = ({f4(scale, 1.0)})
                float4 inputs:bias = ({f4(bias, 0.0)})
                float3 outputs:rgb
            }}

            def Shader "Uv"
            {{
                uniform token info:id = "UsdPrimvarReader_float2"
                string inputs:varname = "st"
                float2 outputs:result
            }}
        }}'''
    return text[: m.start()] + block + text[m.end() :]


def write_variant(name, v, assets=ROOT / "Assets"):
    src, dst = assets / "Room" / SOURCE, assets / "Room" / name
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst, symlinks=True)
    usd = dst / "usd"
    (usd / "lights.usda").write_text(lights_usda(v["panel"], v["temperature"], v["dome"], v["panel_scale"], v["panel_shift"]))
    tex = v.get("textures", {})
    table = set_material((usd / "table.usda").read_text(), "m0", v["table"], v["table_roughness"])
    if "wood" in tex:
        table = add_uvs(table, "Top", tex["wood"], face_varying=True)
        table = set_texture(table, "m0", "/Table/Looks/m0", tex["wood"], v["table"], v["table_roughness"], assets)
    (usd / "table.usda").write_text(table)
    bg = (usd / "background.usda").read_text()
    bg = set_material(bg, "m3", v["cloth"])
    if "weave" in tex:
        bg = add_uvs(bg, "Cloth", tex["weave"], face_varying=False)
        bg = set_texture(bg, "m3", "/Background/Looks/m3", tex["weave"], v["cloth"], None, assets)
    bg = set_material(bg, "m4", v["sleeve"])
    bg = set_material(bg, "m5", v["frame"])
    (usd / "background.usda").write_text(bg)
    (dst / "variant.json").write_text(json.dumps(v, indent=2) + "\n")
    return dst


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prefix", default="school_v3")
    ap.add_argument("--count", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--textured", action="store_true", help="maps on the tabletop and cloth (own stream: variant k "
                    "keeps the lights and colours of the untextured variant k of the same seed)")  # fmt: skip
    ap.add_argument("--v4a", action="store_true", help="textured, colours and maps of RANGES_V4A / *_V4A (school_v4a)")
    args = ap.parse_args()
    args.textured = args.textured or args.v4a
    rng = np.random.default_rng(args.seed)
    for k in range(args.count):
        v = draw(rng, args.v4a)
        if args.textured:
            v["textures"] = draw_textures(np.random.default_rng([args.seed, k, 1]), args.v4a)
        if args.v4a:
            v["profile"] = "v4a"
        write_variant(f"{args.prefix}_{k}", v)
        maps = {s: f"{t['map'].split('/')[1].rsplit('.', 1)[0]} {t['tile']:.2f} m x{t['strength']:.2f}"
                for s, t in v.get("textures", {}).items()}  # fmt: skip
        print(f"{args.prefix}_{k}: panel {v['panel']:.0f} {v['temperature']:.0f} K x{v['panel_scale']:.2f} "
              f"shift {np.round(v['panel_shift'], 2).tolist()} dome {v['dome']:.0f} table {np.round(v['table'], 3).tolist()} "
              f"cloth {np.round(v['cloth'], 3).tolist()}{' ' + str(maps) if maps else ''}")  # fmt: skip


if __name__ == "__main__":
    main()
