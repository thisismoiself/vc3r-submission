#!/usr/bin/env python3
"""Accumulation fly-around video of the office4 stitched reconstruction.

Reveals the per-window predicted point clouds one window at a time (in the
order the frames arrive in the sequence), colored by temporal order, while the
camera slowly orbits the room.  Output: mp4.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import open3d as o3d
import imageio.v2 as imageio
import matplotlib

OUT = Path("/usr/prakt/s0016/vc3r/outputs/replica/stitch_office4")
W_PX, H_PX = 1280, 720


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--src", default="pred", choices=["pred", "oracle", "input"])
    p.add_argument("--mode", default="exterior", choices=["exterior", "interior", "flythrough"],
                   help="exterior=orbit outside; interior=inside looking at table; "
                        "flythrough=reveal from outside then dolly in to survey objects")
    p.add_argument("--dolly-frames", type=int, default=120, help="flythrough: frames for the outside->inside move")
    p.add_argument("--interior-frames", type=int, default=180, help="flythrough: interior survey frames")
    p.add_argument("--total-turns", type=float, default=2.0, help="flythrough: total azimuth turns over the whole shot")
    p.add_argument("--color", default="window", choices=["window", "height"],
                   help="window=temporal stitch order; height=by world Z (reveals table/floor/walls)")
    p.add_argument("--reveal-frames", type=int, default=6, help="video frames per window reveal")
    p.add_argument("--tail-frames", type=int, default=72, help="full-orbit frames at the end")
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--point-size", type=float, default=2.5)
    p.add_argument("--turns", type=float, default=1.0, help="azimuth turns during reveal")
    p.add_argument("--elev-deg", type=float, default=28.0, help="exterior orbit elevation")
    p.add_argument("--flip-up", action="store_true")
    p.add_argument("--subsample", type=int, default=0, help="max pts per window (0=all)")
    # interior-mode framing (world coords; up axis is +Z for Replica)
    p.add_argument("--orbit-radius", type=float, default=2.8, help="interior: eye distance from room center (m)")
    p.add_argument("--eye-z", type=float, default=0.9, help="interior: camera height (m)")
    p.add_argument("--target-z", type=float, default=-0.6, help="interior: look-at height ~ table top (m)")
    p.add_argument("--fov", type=float, default=60.0, help="vertical field of view (deg)")
    return p.parse_args()


def estimate_up(cam_centers, flip):
    c = cam_centers - cam_centers.mean(0)
    _, _, vt = np.linalg.svd(c, full_matrices=False)
    up = vt[-1]  # smallest-variance axis of camera path ~ vertical
    # cameras sit between floor and ceiling; orient up away from the larger
    # (floor) mass is ambiguous, so default to +world and allow --flip-up.
    if up[np.argmax(np.abs(up))] < 0:
        up = -up
    return -up if flip else up


def main():
    a = parse_args()
    d = np.load(OUT / "per_window.npz")
    clouds = d[a.src]                       # [W, Q, 3]
    cam0 = d["cam0"]; f0 = d["f0"]; f1 = d["f1"]
    Wn = clouds.shape[0]
    cam_centers = cam0[:, :3, 3]

    if a.subsample and clouds.shape[1] > a.subsample:
        rng = np.random.default_rng(0)
        idx = rng.choice(clouds.shape[1], a.subsample, replace=False)
        clouds = clouds[:, idx]

    allpts = clouds.reshape(-1, 3)
    center = allpts.mean(0)
    radius = float(np.linalg.norm(allpts - center, axis=1).max())
    up = estimate_up(cam_centers, a.flip_up)
    # build an orbit basis: e1,e2 span the plane perpendicular to up
    tmp = np.array([1.0, 0, 0]) if abs(up[0]) < 0.9 else np.array([0, 1.0, 0])
    e1 = np.cross(up, tmp); e1 /= np.linalg.norm(e1)
    e2 = np.cross(up, e1)
    el = np.deg2rad(a.elev_deg)
    cam_dist = radius * 2.4
    # interior: orbit the camera-path centroid (where the furniture lives), not
    # the cloud mean (which is pulled toward the walls). Eye/target heights are
    # set in world Z (Replica is Z-up: floor ~ -1.3, ceiling ~ +1.6).
    interior_pivot = cam_centers.mean(0).copy()
    interior_pivot[2] = 0.0  # planar pivot; heights applied explicitly below
    target_in = interior_pivot + a.target_z * up
    eye_z_vec = a.eye_z * up

    # renderer + per-window geometries (toggle visibility instead of re-upload)
    r = o3d.visualization.rendering.OffscreenRenderer(W_PX, H_PX)
    r.scene.set_background([1, 1, 1, 1])
    r.scene.scene.set_sun_light([-0.3, -0.5, -0.8], [1, 1, 1], 60000)
    r.scene.scene.enable_sun_light(True)
    r.scene.camera.set_projection(a.fov, W_PX / H_PX, 0.05, 200.0,
                                  o3d.visualization.rendering.Camera.FovType.Vertical)
    mat = o3d.visualization.rendering.MaterialRecord()
    mat.shader = "defaultUnlit"; mat.point_size = a.point_size
    cmap = matplotlib.colormaps["turbo"]
    z_all = allpts[:, 2]
    z_lo, z_hi = np.percentile(z_all, 1), np.percentile(z_all, 99)
    for w in range(Wn):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(clouds[w])
        if a.color == "height":
            t = np.clip((clouds[w][:, 2] - z_lo) / (z_hi - z_lo + 1e-9), 0, 1)
            cols = cmap(t)[:, :3]
        else:
            cols = np.tile(np.array(cmap(w / max(1, Wn - 1))[:3]), (clouds[w].shape[0], 1))
        pcd.colors = o3d.utility.Vector3dVector(cols)
        r.scene.add_geometry(f"w{w}", pcd, mat)
        r.scene.show_geometry(f"w{w}", False)

    # unified camera: orbit `pivot` at horizontal radius hr, camera height ez,
    # looking at a point tz above the pivot (all heights along the up axis).
    pivot = interior_pivot  # camera-path centroid, z=0
    def set_cam(az, hr, ez, tz):
        eye = pivot + hr * (np.cos(az) * e1 + np.sin(az) * e2) + ez * up
        target = pivot + tz * up
        r.scene.camera.look_at(target, eye, up)

    # exterior framing derived from the elevation/zoom used by --mode exterior
    hr_ext, ez_ext, tz_ext = cam_dist * np.cos(el), cam_dist * np.sin(el), float(center[2])
    hr_int, ez_int, tz_int = a.orbit_radius, a.eye_z, a.target_z

    def show_upto(wmax):
        for w in range(Wn):
            r.scene.show_geometry(f"w{w}", w <= wmax)

    frames = []
    if a.mode == "flythrough":
        # Phase A: exterior orbit while windows accumulate.
        nA = Wn * a.reveal_frames
        nB, nC = a.dolly_frames, a.interior_frames
        total = nA + nB + nC
        smooth = lambda u: u * u * (3 - 2 * u)
        for gf in range(total):
            az = 2 * np.pi * a.total_turns * (gf / max(1, total - 1))
            if gf < nA:                                   # reveal, outside
                show_upto(gf // a.reveal_frames)
                hr, ez, tz = hr_ext, ez_ext, tz_ext
            elif gf < nA + nB:                            # dolly inward
                show_upto(Wn - 1)
                s = smooth((gf - nA) / max(1, nB))
                hr = hr_ext + (hr_int - hr_ext) * s
                ez = ez_ext + (ez_int - ez_ext) * s
                tz = tz_ext + (tz_int - tz_ext) * s
            else:                                         # interior survey
                show_upto(Wn - 1)
                hr, ez, tz = hr_int, ez_int, tz_int
            set_cam(az, hr, ez, tz)
            frames.append(np.asarray(r.render_to_image()))
        total_reveal = nA
    else:
        total_reveal = Wn * a.reveal_frames
        hr, ez, tz = (hr_int, ez_int, tz_int) if a.mode == "interior" else (hr_ext, ez_ext, tz_ext)
        for i in range(total_reveal):
            show_upto(i // a.reveal_frames)
            set_cam(2 * np.pi * a.turns * (i / max(1, total_reveal)), hr, ez, tz)
            frames.append(np.asarray(r.render_to_image()))
        show_upto(Wn - 1)
        az0 = 2 * np.pi * a.turns
        for j in range(a.tail_frames):
            set_cam(az0 + 2 * np.pi * (j / a.tail_frames), hr, ez, tz)
            frames.append(np.asarray(r.render_to_image()))

    tag = f"{a.src}_{a.mode}_{a.color}"
    out_mp4 = OUT / f"office4_{tag}_accumulation.mp4"
    imageio.mimsave(out_mp4, frames, fps=a.fps, quality=8, macro_block_size=8)
    imageio.imwrite(OUT / f"office4_{tag}_lastframe.png", frames[total_reveal - 1])
    print(f"[video] {out_mp4}  frames={len(frames)}  {len(frames)/a.fps:.1f}s")
    print(f"[info] up={up.round(3)} center={center.round(2)} radius={radius:.2f} windows={Wn}")


if __name__ == "__main__":
    main()
