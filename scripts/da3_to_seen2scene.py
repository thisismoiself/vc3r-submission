#!/usr/bin/env python3
"""Convert a DA3 Replica room (our 2cm three-state carve cache + clean GT mesh) into a Seen2Scene
scene folder so their `large_scale_completion` (drop_bbox) can complete the DA3 holes.

Per room we emit, under <out-root>/<room>/:
  * fusion_p_1.0_v_0.011.vdb  : GT, an OpenVDB level set of the clean Replica mesh (grid name "tsdf")
  * fusion_p_0.1_v_0.011.vdb  : PARTIAL/src, the DA3 reconstruction upsampled to 1.1cm with the
                                three-state TSDF convention Seen2Scene's loader decodes:
                                   |v| < 0.033*known_ratio  -> BAND   (observed surface)
                                   v >= +0.033              -> EMPTY  (carved free space)
                                   v <= -0.033 (background) -> UNKNOWN(occluded / DA3 hole to fill)
                                Our cache mask OBS/FREE/UNK maps to signed tsdf / +trunc / -trunc(bg),
                                so DA3 holes stay UNKNOWN (NOT free) — otherwise the model fills nothing.
  * meta.json                 : {scene_box, object_bboxes:[1 dummy], object_names:["chair"]}. The box is
                                ignored under drop_bbox but the loader drops scenes with 0 valid objects.

Run with the seen2scene launcher (needs openvdb): s2s_python.sh scripts/da3_to_seen2scene.py --room ...
"""
import os, sys, json, argparse
from pathlib import Path
import numpy as np
import trimesh
import openvdb as vdb

VX = 0.011
TRUNC = 3 * VX          # 0.033
HALF = 3                # level-set half width in voxels
BAND = 0.85 * TRUNC     # OBS clip target, safely < TRUNC*known_ratio(0.999) -> classified BAND


def gt_grid(mesh_path):
    m = trimesh.load(mesh_path, process=False)
    V = np.asarray(m.vertices, np.float32)
    F = np.asarray(m.faces, np.int32)
    xform = vdb.createLinearTransform(voxelSize=VX)
    g = vdb.FloatGrid.createLevelSetFromPolygons(V, triangles=F, transform=xform, halfWidth=HALF)
    g.name = "tsdf"
    return g, m.bounds


def partial_grid(cache_path):
    """Upsample our 2cm three-state DA3 carve to a 1.1cm signed-TSDF level set with UNKNOWN background."""
    c = np.load(cache_path)
    mask = c["mask"]                       # 0=OBS,1=FREE,2=UNK  (Voxel OBS/FREE/UNK)
    pt = c["partial_tsdf"].astype(np.float32)
    o = c["origin"].astype(np.float64)     # world coord of 2cm voxel [0,0,0]
    v2 = float(c["voxel"])                 # 0.02
    OBS, FREE, UNK = 0, 1, 2

    # output 1.1cm index range that covers the cache world extent (index i -> world i*VX, matching GT xform)
    wmin = o
    wmax = o + np.array(mask.shape) * v2
    i0 = np.floor(wmin / VX).astype(int)
    i1 = np.ceil(wmax / VX).astype(int)
    shp = (i1 - i0).tolist()

    # world center of each output voxel -> nearest 2cm cache index
    gi = [np.arange(i0[d], i1[d]) for d in range(3)]
    WX = (gi[0] * VX)[:, None, None]
    WY = (gi[1] * VX)[None, :, None]
    WZ = (gi[2] * VX)[None, None, :]
    ci = np.floor((WX - o[0]) / v2).astype(int)
    cj = np.floor((WY - o[1]) / v2).astype(int)
    ck = np.floor((WZ - o[2]) / v2).astype(int)
    ci = np.clip(ci, 0, mask.shape[0] - 1)
    cj = np.clip(cj, 0, mask.shape[1] - 1)
    ck = np.clip(ck, 0, mask.shape[2] - 1)
    smask = mask[ci, cj, ck]               # broadcast gather -> [shp]
    stsdf = pt[ci, cj, ck]

    dense = np.full(shp, -TRUNC, np.float32)          # default UNKNOWN (negative background)
    dense[smask == FREE] = TRUNC                       # EMPTY (carved free space)
    obs = smask == OBS
    dense[obs] = np.clip(stsdf[obs], -BAND, BAND)      # BAND (observed surface), guaranteed |v|<trunc

    g = vdb.FloatGrid(-TRUNC)                           # background = UNKNOWN
    g.copyFromArray(dense, ijk=tuple(int(x) for x in i0), tolerance=1e-6)  # activate non-bg (OBS+FREE)
    g.transform = vdb.createLinearTransform(voxelSize=VX)
    g.name = "tsdf"
    counts = {n: int((smask == k).sum()) for n, k in [("OBS", OBS), ("FREE", FREE), ("UNK", UNK)]}
    return g, counts, g.activeVoxelCount()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--room", required=True)
    ap.add_argument("--cache-dir", default="outputs/tsdf_cache_replica_da3_006")
    ap.add_argument("--mesh-root", default="/usr/prakt/s0016/Replica")
    ap.add_argument("--out-root", default="/usr/prakt/s0016/seen2scene_eval/da3_replica_scenes")
    ap.add_argument("--category", default="chair")
    args = ap.parse_args()

    out = Path(args.out_root) / args.room
    out.mkdir(parents=True, exist_ok=True)

    gg, bounds = gt_grid(f"{args.mesh_root}/{args.room}_mesh.ply")
    vdb.write(str(out / "fusion_p_1.0_v_0.011.vdb"), grids=[gg])

    pg, counts, nact = partial_grid(f"{args.cache_dir}/replica_{args.room}.npz")
    vdb.write(str(out / "fusion_p_0.1_v_0.011.vdb"), grids=[pg])

    lo, hi = bounds[0].tolist(), bounds[1].tolist()
    ctr = [(lo[i] + hi[i]) / 2 for i in range(3)]
    dummy = [[ctr[0] - 0.25, ctr[1] - 0.25, lo[2]], [ctr[0] + 0.25, ctr[1] + 0.25, lo[2] + 0.5]]
    meta = {"scene_box": [lo, hi], "object_bboxes": [dummy], "object_names": [args.category]}
    json.dump(meta, open(out / "meta.json", "w"))

    print(f"{args.room}: GT lvlset {gg.activeVoxelCount():,} act | partial act {nact:,} "
          f"(OBS {counts['OBS']:,} FREE {counts['FREE']:,} UNK {counts['UNK']:,}) -> {out}")


if __name__ == "__main__":
    main()
