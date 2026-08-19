"""Convert a Wavefront .obj mesh to MuJoCo's legacy binary .msh format.

Why: mujoco-py (MuJoCo 2.1) can only load .stl or .msh meshes, and STL has no
texture coordinates -- converting through STL would strip the UVs that a color
texture (e.g. 8_quart_pot_color.png) is mapped with. The legacy .msh format
carries per-vertex texcoords, so a textured .obj survives the trip.

Format (all little-endian):
    int32   nvertex, nnormal, ntexcoord, nface
    float32 vertex[3*nvertex]
    float32 normal[3*nnormal]          (nnormal must be 0 or nvertex)
    float32 texcoord[2*ntexcoord]      (ntexcoord must be 0 or nvertex)
    int32   face[3*nface]

OBJ faces index positions/UVs/normals independently, so each unique
(v, vt, vn) triple becomes one unified output vertex. Quads (and larger
polygons) are fan-triangulated. The texture V coordinate is flipped
(v -> 1-v): MuJoCo samples images top-down while OBJ UVs are bottom-up.

Usage:
    python obj2msh.py input.obj output.msh
"""
import argparse
import struct
import sys


def convert(obj_path, msh_path):
    positions, uvs, normals = [], [], []
    verts = {}          # (v_idx, vt_idx, vn_idx) -> unified vertex index
    out_v, out_uv, out_n, faces = [], [], [], []

    def unified(token):
        idx = tuple(int(p) - 1 if p else -1 for p in (token.split("/") + ["", ""])[:3])
        if idx not in verts:
            verts[idx] = len(out_v)
            out_v.append(positions[idx[0]])
            if idx[1] >= 0:
                u, v = uvs[idx[1]]
                out_uv.append((u, 1.0 - v))
            if idx[2] >= 0:
                out_n.append(normals[idx[2]])
        return verts[idx]

    with open(obj_path) as f:
        for line in f:
            parts = line.split()
            if not parts:
                continue
            if parts[0] == "v":
                positions.append(tuple(float(x) for x in parts[1:4]))
            elif parts[0] == "vt":
                uvs.append(tuple(float(x) for x in parts[1:3]))
            elif parts[0] == "vn":
                normals.append(tuple(float(x) for x in parts[1:4]))
            elif parts[0] == "f":
                corners = [unified(t) for t in parts[1:]]
                for i in range(1, len(corners) - 1):   # fan-triangulate
                    faces.append((corners[0], corners[i], corners[i + 1]))

    nv = len(out_v)
    # msh requires normals/texcoords to cover all vertices or be absent
    if out_uv and len(out_uv) != nv:
        print(f"WARNING: only {len(out_uv)}/{nv} vertices have UVs; dropping texcoords")
        out_uv = []
    if out_n and len(out_n) != nv:
        print(f"WARNING: only {len(out_n)}/{nv} vertices have normals; dropping normals")
        out_n = []

    with open(msh_path, "wb") as f:
        f.write(struct.pack("<4i", nv, len(out_n), len(out_uv), len(faces)))
        for arr, fmt in ((out_v, "<3f"), (out_n, "<3f"), (out_uv, "<2f")):
            for item in arr:
                f.write(struct.pack(fmt, *item))
        for face in faces:
            f.write(struct.pack("<3i", *face))

    print(f"{msh_path}: {nv} verts, {len(out_n)} normals, {len(out_uv)} texcoords, {len(faces)} tris")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("obj", help="input .obj path")
    ap.add_argument("msh", help="output .msh path")
    args = ap.parse_args()
    sys.exit(convert(args.obj, args.msh))
