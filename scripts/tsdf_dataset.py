#!/usr/bin/env python3
"""Cached multi-crop dataloader for Stage-1 TSDF completion.

Design (keeps training GPU-bound):
  * PRECOMPUTE once per scene (the expensive, trajectory-independent part): render depth along
    the trajectory -> partial cloud; build a FULL-SCENE voxel grid; compute GT TSDF, the
    observed/free/unknown mask, and the partial TSDF over it; save to a small npz (~20-40 MB).
  * AT TRAIN TIME (cheap): sample random 128^3 crops biased to occlusion frontiers
    (unknown voxels adjacent to observed surface), with 90deg-rotation / flip augmentation.

Generic over (mesh, poses, intrinsics), so Replica and SCRREAM use the same code.
"""
import argparse, sys, time
from pathlib import Path
import numpy as np
import open3d as o3d
import torch
import trimesh
from scipy.spatial import cKDTree
from scipy.ndimage import binary_dilation, label

REPO = Path("/usr/prakt/s0016/vc3r")
sys.path.insert(0, str(REPO / "scripts"))
from tsdf_data_pipeline import render_depth, load_poses, load_intrinsics   # noqa: E402

VOXEL, BAND, GRID = 0.02, 0.10, 128
OBSERVED, FREE, UNKNOWN = 0, 1, 2


def precompute_scene(mesh_path, poses_path, intr_path, out_npz, n_frames=40,
                     voxel=VOXEL, band=BAND, max_partial=500_000):
    """Build and cache the full-scene TSDF grids for one scene (loads mesh+poses from disk)."""
    K, W, H = load_intrinsics(intr_path)
    tm = trimesh.load(str(mesh_path), force="mesh", process=False, skip_materials=True)
    verts = np.asarray(tm.vertices, np.float32); faces = np.asarray(tm.faces, np.uint32)
    poses = load_poses(poses_path)
    name = Path(mesh_path).parent.name
    return precompute_from_mesh(verts, faces, poses, K, W, H, out_npz, n_frames=n_frames,
                                voxel=voxel, band=band, max_partial=max_partial, name=name)


def precompute_from_mesh(verts, faces, poses, K, W, H, out_npz, n_frames=40,
                         voxel=VOXEL, band=BAND, max_partial=500_000, name="scene", kdt_workers=-1,
                         bounds=None, max_vox=60_000_000, free_flood=False,
                         ext_depths=None, ext_K=None, ext_partial=None):
    """Core: given an in-memory mesh + camera trajectory, build & cache the full-scene TSDF grids.
    Shared by SCRREAM (disk meshes) and 3D-FRONT (assembled rooms). Renders depth along the
    trajectory -> partial cloud, carves observed/free/unknown, computes GT + partial TSDF.

    `bounds`=(lo,hi) overrides the grid extent (use a ROBUST box, e.g. the floor footprint, so a
    single outlier vertex can't inflate the grid to billions of voxels). `max_vox` is a hard guard:
    rooms whose grid would exceed it are skipped rather than OOM-ing the machine."""
    verts = np.ascontiguousarray(verts, np.float32); faces = np.ascontiguousarray(faces, np.uint32)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.core.Tensor(verts), o3d.core.Tensor(faces))
    fids = np.unique(np.linspace(0, len(poses) - 1, min(n_frames, len(poses))).round().astype(int))
    t0 = time.time()

    # DEPTH SOURCE. Default: raycast the (clean) mesh -> partial = perfect visibility carve.
    # ext_depths (e.g. DA3-predicted depth, aligned 1:1 with `fids`): use the EXTERNAL depth for the
    # partial cloud + free/observed carving, so the partial reflects the real reconstruction (with
    # its holes/noise), while GT below still comes from the clean mesh. ext_K / ext_partial give the
    # external camera intrinsics and pre-back-projected world points.
    if ext_depths is not None:
        depths = list(ext_depths)
        cams = [poses[f][:3, 3] for f in fids]
        partial = np.ascontiguousarray(ext_partial, np.float64)
        Kc = np.asarray(ext_K, np.float64); Hc, Wc = int(depths[0].shape[0]), int(depths[0].shape[1])
    else:
        depths, cams, parts = [], [], []
        for f in fids:
            d, hits = render_depth(scene, K, poses[f], W, H)
            depths.append(d); cams.append(poses[f][:3, 3]); parts.append(hits)
        partial = np.concatenate(parts, 0).astype(np.float64)
        Kc, Wc, Hc = K, W, H
    if len(partial) < 1000:
        print(f"[precompute] {name}: only {len(partial)} partial pts — skipping (empty room / bad traj)",
              flush=True)
        return None
    if len(partial) > max_partial:
        partial = partial[np.random.default_rng(0).choice(len(partial), max_partial, replace=False)]

    # grid extent: caller-supplied robust box, else mesh bounds (+ small margin), aligned to voxel
    if bounds is not None:
        lo = np.asarray(bounds[0], np.float64) - 3 * voxel; hi = np.asarray(bounds[1], np.float64) + 3 * voxel
    else:
        lo = verts.min(0) - 3 * voxel; hi = verts.max(0) + 3 * voxel
    dims = np.ceil((hi - lo) / voxel).astype(int)
    if int(np.prod(dims.astype(np.int64))) > max_vox:
        print(f"[precompute] {name}: grid {dims.tolist()} = {np.prod(dims)/1e6:.0f}M vox > "
              f"{max_vox/1e6:.0f}M cap — skipping (outlier geometry?)", flush=True)
        return None
    lin = [((np.arange(dims[i]) + 0.5) * voxel + lo[i]).astype(np.float32) for i in range(3)]
    gx, gy, gz = np.meshgrid(*lin, indexing="ij")
    vflat = np.stack([gx, gy, gz], -1).reshape(-1, 3)             # float32: halves memory traffic
    D0, D1, D2 = dims

    tg = time.time()
    gt = np.clip(scene.compute_signed_distance(o3d.core.Tensor(vflat)).numpy(),
                 -band, band).reshape(dims)
    tsdf_t = time.time() - tg

    # visibility carving. Kept lean because it is memory-bandwidth bound (the dominant cost when
    # several workers run at once): float32 throughout, no big transposes, in-place where possible.
    tc = time.time()
    obs = np.zeros(len(vflat), bool); free = np.zeros(len(vflat), bool); eps = np.float32(1.5 * voxel)
    fx, fy = np.float32(Kc[0, 0]), np.float32(Kc[1, 1]); cx, cy = np.float32(Kc[0, 2]), np.float32(Kc[1, 2])
    for f, d in zip(fids, depths):
        w2c = np.linalg.inv(poses[f]).astype(np.float32)
        vc = vflat @ w2c[:3, :3].T + w2c[:3, 3]                   # (N,3)@(3,3): no transpose of vflat
        z = vc[:, 2]; u = fx * vc[:, 0] / z + cx; v = fy * vc[:, 1] / z + cy
        inb = (z > 0) & (u >= 0) & (u < Wc) & (v >= 0) & (v < Hc)
        ui = np.clip(u, 0, Wc - 1).astype(np.int32); vi = np.clip(v, 0, Hc - 1).astype(np.int32)
        dref = d[vi, ui]; valid = inb & np.isfinite(dref)
        obs |= valid & (np.abs(z - dref) < eps); free |= valid & (z < dref - eps)
    mask = np.full(len(vflat), UNKNOWN, np.int8); mask[free] = FREE; mask[obs] = OBSERVED
    mask = mask.reshape(dims)
    # exterior -> FREE: flood empty space through everything that is NOT an observed surface,
    # seeded from the carved FREE voxels AND the grid boundary (a tight footprint+margin box, so
    # its outer shell is genuinely outside the room). Any UNKNOWN reachable that way is empty space
    # the cameras simply never rayed through, not a real occlusion -> relabel it FREE so the model
    # is neither asked to fill it nor (with RePaint replace) allowed to hallucinate into it. What
    # survives as UNKNOWN is then only pockets enclosed by observed surface = true occlusion.
    if free_flood:
        passable = mask != OBSERVED
        lbl, _ = label(passable)                                   # 6-connectivity components
        seed = (mask == FREE)
        seed[0, :, :] = seed[-1, :, :] = True; seed[:, 0, :] = seed[:, -1, :] = True
        seed[:, :, 0] = seed[:, :, -1] = True
        seed &= passable
        outside_labels = np.unique(lbl[seed]); outside_labels = outside_labels[outside_labels != 0]
        newly_free = np.isin(lbl, outside_labels) & (mask == UNKNOWN)
        mask[newly_free] = FREE
    carve_t = time.time() - tc

    # partial TSDF (normal-signed). KD-tree query is multi-threaded (workers=-1); we only need
    # the signed distance within the truncation band, so query just the near-surface voxels — the
    # far ones are saturated to +band anyway. This is the dominant cost, so it matters a lot.
    tp = time.time()
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(partial))
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    pcd.orient_normals_towards_camera_location(np.mean(cams, 0))
    pn = np.asarray(pcd.normals); tree = cKDTree(partial)
    ptsdf = np.full(len(vflat), band, np.float32)                  # default: empty / far
    near = np.abs(gt.reshape(-1)) < band                          # only voxels near ANY surface
    dist, idx = tree.query(vflat[near], k=1, workers=kdt_workers)
    sgn = np.sign(np.einsum("ij,ij->i", vflat[near] - partial[idx], pn[idx])); sgn[sgn == 0] = 1
    ptsdf[near] = np.clip(sgn * dist, -band, band)
    ptsdf = ptsdf.reshape(dims).astype(np.float16)
    part_t = time.time() - tp

    Path(out_npz).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_npz, gt_tsdf=gt.astype(np.float16), partial_tsdf=ptsdf,
                        mask=mask, origin=lo, voxel=voxel, dims=dims, band=np.float32(band))
    nun = int((mask == UNKNOWN).sum())
    print(f"[precompute] {name}: dims={dims.tolist()} "
          f"({mask.size/1e6:.1f}M vox) unknown={100*nun/mask.size:.0f}%  "
          f"partial={len(partial):,}  {time.time()-t0:.0f}s "
          f"(sdf={tsdf_t:.0f} carve={carve_t:.0f} part={part_t:.0f}) -> {Path(out_npz).name}", flush=True)
    return out_npz


def da3_corrupt(pt, mask, rng, band, voxel, noise_vox=1.8, n_holes=60, hole_r=6):
    """Make a CLEAN (mesh-rendered) partial look like a DA3 reconstruction, so 3D-FRONT training
    teaches DA3-hole-filling. Calibrated to the measured DA3 profile (obs->GT noise ~0.4-1.7 vox;
    missing-surface holes -> more unknown):
      1) surface JITTER: Gaussian noise (~noise_vox) on the near-surface partial TSDF -> the observed
         surface is displaced from the true surface, exactly like DA3's depth error.
      2) surface DROPOUT: remove patches centred ON the observed surface (where DA3 'failed to
         reconstruct') -> those observed/free voxels become UNKNOWN (new targets) and the partial
         goes empty there. Random volume boxes would only delete air, so holes are surface-seeded.
    GT is untouched (still the clean mesh). Operates in-place on copies."""
    pt = pt.copy().astype(np.float32); mask = mask.copy()
    surf = 0.7 * voxel
    obs_surf = np.argwhere((mask == OBSERVED) & (np.abs(pt) < surf))     # BEFORE jitter
    near = np.abs(pt) < band
    pt[near] += rng.normal(0, noise_vox * voxel, int(near.sum())).astype(np.float32)
    np.clip(pt, -band, band, out=pt)
    D = mask.shape
    if len(obs_surf):
        ctr = obs_surf[rng.choice(len(obs_surf), min(n_holes, len(obs_surf)), replace=False)]
        for c in ctr:
            r = int(rng.integers(hole_r // 2, hole_r + 4))
            sl = tuple(slice(max(0, c[a] - r), min(D[a], c[a] + r)) for a in range(3))
            mb = mask[sl]; holed = (mb == OBSERVED) | (mb == FREE)       # views -> in-place
            mb[holed] = UNKNOWN; sub = pt[sl]; sub[holed] = band
    return pt, mask


def _crop(arr, c, size, fill):
    """Extract a size^3 crop centred at index c, padding out-of-bounds with `fill`."""
    out = np.full((size, size, size), fill, arr.dtype)
    lo = c - size // 2
    a0 = np.maximum(lo, 0); a1 = np.minimum(lo + size, arr.shape)
    o0 = a0 - lo; o1 = o0 + (a1 - a0)
    out[o0[0]:o1[0], o0[1]:o1[1], o0[2]:o1[2]] = arr[a0[0]:a1[0], a0[1]:a1[1], a0[2]:a1[2]]
    return out


class TSDFCropDataset(torch.utils.data.Dataset):
    """Random occlusion-frontier crops from cached full-scene grids, with grid augmentation."""
    def __init__(self, npzs, grid=GRID, band=BAND, samples_per_epoch=2000, seed=0, da3_sim=False,
                 pool_size=None, refresh_every=150):
        """pool_size None (or >= len(npzs)) = load ALL rooms into RAM (original behaviour). A smaller
        pool_size STREAMS: keep only `pool_size` rooms resident and swap one out for a fresh one off
        disk every `refresh_every` samples, so training cycles through arbitrarily many cached rooms
        (e.g. all 12k) at bounded RAM. Each DataLoader worker keeps its own pool, so peak RAM is
        ~workers*pool_size rooms -- size the sbatch accordingly."""
        self.grid, self.band, self.n, self.da3_sim = grid, band, samples_per_epoch, da3_sim
        self.paths = list(npzs)
        self.rng = np.random.default_rng(seed)
        self.refresh_every = refresh_every
        self.count = 0
        n_all = len(self.paths)
        self.pool_size = n_all if not pool_size else min(pool_size, n_all)
        self.streaming = self.pool_size < n_all
        self.pool_idx = []
        self.scenes = {}
        self._init_pool(self.pool_size)                        # skips unreadable/empty rooms
        tag = (f"STREAM pool={len(self.scenes)}/{n_all} refresh_every={refresh_every}"
               if self.streaming else f"full-load {len(self.scenes)}/{n_all} rooms")
        print(f"[dataset] {tag}", flush=True)

    def _load(self, p):
        try:
            d = np.load(p)                                     # copy out of the mmap so the fd can close
            gt, pt, mask = np.array(d["gt_tsdf"]), np.array(d["partial_tsdf"]), np.array(d["mask"])
            frontier = (mask == UNKNOWN) & binary_dilation(mask == OBSERVED, iterations=2)
            cand = np.argwhere(frontier)
            if len(cand) == 0:
                cand = np.argwhere(mask == UNKNOWN)
            if len(cand) == 0:
                return None                                    # unusable room -> skip
            return {"gt": gt, "pt": pt, "mask": mask, "cand": cand}
        except Exception as e:                                 # a corrupt/half-written npz must never hang the run
            print(f"[dataset] skip {Path(p).name}: {e}", flush=True)
            return None

    def _init_pool(self, want):
        """Fill the resident pool with `want` VALID rooms (retrying past unreadable ones)."""
        order = list(range(len(self.paths)))
        self.rng.shuffle(order)
        for i in order:
            if len(self.scenes) >= want:
                break
            s = self._load(self.paths[i])
            if s is not None:
                self.scenes[i] = s; self.pool_idx.append(i)

    def _maybe_refresh(self):
        if not self.streaming:
            return
        self.count += 1
        if self.count % self.refresh_every:
            return
        resident = set(self.pool_idx)
        newi = int(self.rng.integers(len(self.paths))); tries = 0
        while newi in resident and tries < 50:
            newi = int(self.rng.integers(len(self.paths))); tries += 1
        if newi in resident:
            return
        s = self._load(self.paths[newi])
        if s is None:                                          # bad room: keep current pool, try again next time
            return
        evict = self.pool_idx[int(self.rng.integers(len(self.pool_idx)))]
        self.scenes.pop(evict, None); self.pool_idx.remove(evict)
        self.scenes[newi] = s; self.pool_idx.append(newi)

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        self._maybe_refresh()
        key = self.pool_idx[int(self.rng.integers(len(self.pool_idx)))]
        s = self.scenes[key]
        c = s["cand"][self.rng.integers(len(s["cand"]))]
        gt = _crop(s["gt"].astype(np.float32), c, self.grid, self.band)      # outside = +band (empty)
        pt = _crop(s["pt"].astype(np.float32), c, self.grid, self.band)
        mask = _crop(s["mask"], c, self.grid, UNKNOWN)                        # outside = unknown
        # augmentation: 90deg rotations in the horizontal plane (axes 0,1) + flips (grid-exact)
        k = self.rng.integers(4)
        if k: gt, pt, mask = (np.rot90(a, k, (0, 1)).copy() for a in (gt, pt, mask))
        for ax in (0, 1):
            if self.rng.random() < 0.5:
                gt, pt, mask = (np.flip(a, ax).copy() for a in (gt, pt, mask))
        if self.da3_sim:                                  # corrupt clean partial -> DA3-like (GT stays clean)
            pt, mask = da3_corrupt(pt, mask, self.rng, self.band, VOXEL)
        # 5-channel input: partial tsdf (normalized) + mask one-hot + confidence
        pt_n = torch.from_numpy(pt) / self.band
        oh = torch.nn.functional.one_hot(torch.from_numpy(mask.astype(np.int64)), 3).permute(3, 0, 1, 2).float()
        inp = torch.cat([pt_n[None], oh, torch.ones(1, *pt.shape)], 0)
        return inp, torch.from_numpy(gt)[None], torch.from_numpy(mask.astype(np.int64))[None]


def _pad_center(arr, size, fill):
    """Center a (<=size)^3 room inside a size^3 grid, padding the rest with `fill`."""
    out = np.full((size, size, size), fill, arr.dtype)
    s = [min(size, arr.shape[a]) for a in range(3)]
    off = [(size - s[a]) // 2 for a in range(3)]
    out[off[0]:off[0]+s[0], off[1]:off[1]+s[1], off[2]:off[2]+s[2]] = arr[:s[0], :s[1], :s[2]]
    return out


class CoarseRoomDataset(torch.utils.data.Dataset):
    """Cascade COARSE stage: whole-room completion at coarse (8cm) resolution. Each item is the FULL
    downsampled room centered in a grid^3 volume (no cropping) -> the coarse model sees the entire
    room in one pass, so it has global context and completes wall/floor/ceiling planes by construction.
    Same 5-channel format as TSDFCropDataset so it drops into the existing training loop."""
    def __init__(self, npzs, grid=GRID, band=BAND, samples_per_epoch=2000, seed=0):
        self.grid, self.band, self.n = grid, band, samples_per_epoch
        self.scenes = []
        for p in npzs:
            d = np.load(p); mask = d["mask"]
            if max(mask.shape) > grid:
                continue                                          # room too big for the coarse grid
            self.scenes.append({"gt": d["gt_tsdf"], "pt": d["partial_tsdf"], "mask": mask})
        print(f"[coarse] {len(self.scenes)} whole rooms fit in {grid}^3 (of {len(npzs)})", flush=True)
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        s = self.scenes[self.rng.integers(len(self.scenes))]
        gt = _pad_center(s["gt"].astype(np.float32), self.grid, self.band)
        pt = _pad_center(s["pt"].astype(np.float32), self.grid, self.band)
        mask = _pad_center(s["mask"], self.grid, UNKNOWN)
        k = self.rng.integers(4)
        if k: gt, pt, mask = (np.rot90(a, k, (0, 1)).copy() for a in (gt, pt, mask))
        for ax in (0, 1):
            if self.rng.random() < 0.5:
                gt, pt, mask = (np.flip(a, ax).copy() for a in (gt, pt, mask))
        pt_n = torch.from_numpy(pt) / self.band
        oh = torch.nn.functional.one_hot(torch.from_numpy(mask.astype(np.int64)), 3).permute(3, 0, 1, 2).float()
        inp = torch.cat([pt_n[None], oh, torch.ones(1, *pt.shape)], 0)
        return inp, torch.from_numpy(gt)[None], torch.from_numpy(mask.astype(np.int64))[None]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--scene-dir", default="/usr/prakt/s0016/SCRREAM/dataset/scene01")
    ap.add_argument("--traj", default="scene01_full_00")
    ap.add_argument("--cache", default="/usr/prakt/s0016/vc3r/outputs/tsdf_cache/scene01.npz")
    ap.add_argument("--n-frames", type=int, default=40)
    args = ap.parse_args()

    sd = Path(args.scene_dir)
    if not Path(args.cache).exists():
        precompute_scene(sd / f"{sd.name}_mesh.ply", sd / args.traj / "camera_pose",
                         sd / args.traj / "intrinsics.txt", args.cache, n_frames=args.n_frames)

    if args.smoke:
        from stage1_unet import OcclusionUNet, masked_surface_l1
        ds = TSDFCropDataset([args.cache], samples_per_epoch=64)
        dl = torch.utils.data.DataLoader(ds, batch_size=2, num_workers=2)
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        net = OcclusionUNet(in_ch=5, base=16, band=BAND).to(dev)
        opt = torch.optim.AdamW(net.parameters(), lr=1e-3)
        print(f"[smoke] device={dev}  streaming crops through the UNet...")
        t0 = time.time()
        for b, (inp, gt, mask) in enumerate(dl):
            inp, gt, mask = inp.to(dev), gt.to(dev), mask.to(dev)
            unk = (mask == UNKNOWN).float().mean().item()
            pred = net(inp); loss = masked_surface_l1(pred, gt, mask, BAND - VOXEL)
            opt.zero_grad(); loss.backward(); opt.step()
            print(f"  batch {b} inp={tuple(inp.shape)} unknown-frac={unk:.2f} loss={loss.item()*100:.2f} "
                  f"t={time.time()-t0:.1f}s", flush=True)
            if b >= 4:
                break
        print("[smoke] PASS — cached crops feed the UNet and train.", flush=True)


if __name__ == "__main__":
    main()
