"""Full-scene reconstruction of office4 (the held-out validation room).

Tiles the office4 trajectory with consecutive non-overlapping 8-frame windows,
regenerates l13 DA3 tokens EXACTLY as in training (extract_da3_tokens, layers
[1,3], max_tokens 2048, image 392x518), runs adapter -> NOVA3R decode per
window, places each window in the shared world frame using the metric
norm_factor recovered by the NOVA3R encoder, concatenates all windows, and
evaluates the stitched cloud against the office4 GT.

For each window we also decode the NOVA3R encoder tokens directly (single-seed
z) -> that is the autoencoder ceiling ("oracle") under identical conditioning.
"""
from __future__ import annotations

import os
import sys
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import trimesh
from omegaconf import OmegaConf
from PIL import Image
from scipy.spatial import cKDTree

os.environ.setdefault("DA3_LOG_LEVEL", "WARN")
REPO = Path(__file__).resolve().parents[1]
NOVA = REPO / "nova3r"
for p in [str(REPO), str(NOVA / "third_party"), str(NOVA), str(REPO / "da3" / "src")]:
    if p not in sys.path:
        sys.path.insert(0, p)

from vc3r.alignment import DA3ToNOVA3RAlignment                         # noqa: E402
from nova3r.models.model_wrapper import BatchModelWrapper                # noqa: E402
from nova3r.flow_matching.solver import ODESolver                        # noqa: E402
from nova3r.inference import normalize_input                            # noqa: E402
from vc3r.replica import crop_visible_world_points, project_world_points  # noqa: E402
from vc3r.artifacts import DA3_MODEL_ID, DA3_REVISION, resolve_da3_snapshot  # noqa: E402
from vc3r.runtime import (                                              # noqa: E402
    PointFlowCorrector,
    extract_da3_tokens,
    integrate_point_corrector,
    load_da3_model,
    load_nova3r_model,
    to_exact,
    world_to_first_camera,
)

NOVA_CKPT = REPO / "checkpoints" / "nova3r" / "scene_ae" / "checkpoint-last.pth"
ROOM = "office4"
REPLICA_ROOT = REPO / "datasets" / "replica"
OUTPUT_ROOT = REPO / "outputs" / "replica"
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")

K_SAMPLE = 8192
DA3_LAYERS = [1, 3]
DA3_MAX_TOKENS = 2048
IMG_H, IMG_W = 392, 518


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=Path, required=True, help="adapter checkpoint to evaluate")
    p.add_argument("--nova-ckpt", type=Path, default=NOVA_CKPT,
                   help="NOVA3R scene-AE checkpoint (requires adjacent .hydra/config.yaml)")
    p.add_argument("--room", default="office4", help="room to reconstruct (e.g. office0, room1, breakfast_room)")
    p.add_argument("--data-root", type=Path, default=REPLICA_ROOT,
                   help="dataset root in Replica layout (default: <repo>/datasets/replica)")
    p.add_argument("--sequence-dir", type=Path, default=None,
                   help="Optional scene/sequence directory containing traj.txt and results/. "
                        "Defaults to <data-root>/<room>.")
    p.add_argument("--gt-mesh", type=Path, default=None,
                   help="Optional GT mesh path. Defaults to <data-root>/<room>_mesh.ply.")
    p.add_argument("--gt-cloud", type=Path, default=None,
                   help="Optional fixed GT point cloud for scoring. Defaults to the repository's "
                        "office4-style GT cloud when present, otherwise samples --gt-mesh.")
    p.add_argument("--output-root", type=Path, default=OUTPUT_ROOT,
                   help="parent directory for stitch_<room>_<tag> outputs "
                        "(default: <repo>/outputs/replica)")
    p.add_argument("--da3-model", default=DA3_MODEL_ID,
                   help="DA3 Hugging Face model ID or local snapshot directory")
    p.add_argument("--da3-revision", default=DA3_REVISION,
                   help="exact Hugging Face revision used when --da3-model is a model ID")
    p.add_argument("--offline", action="store_true",
                   help="use only local DA3 files; never access Hugging Face over the network")
    p.add_argument("--gtfree-tokens", action="store_true",
                   help="Extract DA3 tokens with cam_token=None (no GT poses), matching a "
                        "GT-free-input adapter. Placement/scale still use GT (registration is separate).")
    p.add_argument("--da3pose-tokens", action="store_true",
                   help="Extract DA3 tokens with cam_token from DA3-PREDICTED poses (matches "
                        "--pose-source da3pred training). Placement/scale still use GT.")
    p.add_argument("--complete-target", action="store_true",
                   help="Build the encode/pts_norm pool from the COMPLETE frustum crop (amodal, no "
                        "occlusion cull) instead of the visible surface. Use with adapters trained on "
                        "complete targets; makes the oracle and conditioning amodal like NOVA3R's src_complete.")
    p.add_argument("--stride", type=int, default=10, help="intra-window frame stride (in-distribution: 10)")
    p.add_argument("--n-frames", type=int, default=8)
    p.add_argument("--num-queries", type=int, default=8192, help="decoded points per window")
    p.add_argument("--importance", action="store_true",
                   help="Adaptive importance-sampled decode: densify fine geometry (chair legs, "
                        "concave interiors) by resampling the FM prior around inits that produced "
                        "sparse (high-detail) output points. ODE-correct (stays on the uniform path).")
    p.add_argument("--imp-extra", type=int, default=8192,
                   help="Extra importance queries per window (added on top of --num-queries).")
    p.add_argument("--imp-frac", type=float, default=0.35,
                   help="Top fraction of pass-1 points (by inverse local density) whose inits seed pass 2.")
    p.add_argument("--imp-sigma", type=float, default=0.03,
                   help="Std of the Gaussian init perturbation in the [-1,1] prior frame.")
    p.add_argument("--filter-sor", action="store_true",
                   help="Statistical outlier removal on the stitched pred/oracle clouds before eval "
                        "(prunes floating decode noise that hurts precision).")
    p.add_argument("--sor-k", type=int, default=16, help="k neighbours for --filter-sor.")
    p.add_argument("--sor-z", type=float, default=2.0, help="z-threshold (mean+z*std) for --filter-sor.")
    p.add_argument("--fm-sampling", default=None, help="ODE solver override (e.g. midpoint, euler)")
    p.add_argument("--fm-step-size", type=float, default=None, help="ODE step size override (smaller=finer)")
    p.add_argument("--max-windows", type=int, default=None)
    p.add_argument("--out-tag", default="", help="suffix for output dir (e.g. run1_s5)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--decoder-ckpt", type=Path, default=None,
                   help="Optional fine-tuned pts3d_head (decoder) state_dict to load into NOVA3R "
                        "before eval (diagnostic: measures the FT decoder's oracle in world metric). "
                        "Affects both oracle and pred decode; for oracle numbers the adapter is irrelevant.")
    p.add_argument("--point-corrector", type=Path, default=None,
                   help="Optional Stage-2 point-flow corrector checkpoint.")
    p.add_argument("--corrector-steps", type=int, default=6, help="Corrector integration steps.")
    p.add_argument("--corrector-steps-sweep", type=str, default=None,
                   help="Comma list of corrector step counts to evaluate in ONE run (decodes each "
                        "window once, applies the corrector at each step count). 0 = uncorrected "
                        "baseline. E.g. '0,6,12,20'. Reports FURN F@2 per step count.")
    p.add_argument("--geom-points", type=int, default=512,
                   help="geometry points fed to a geom-conditioned adapter (must match training).")
    p.add_argument("--geom-source", choices=["pts_norm", "da3"], default="pts_norm",
                   help="geometry stream for a geom-conditioned adapter: 'pts_norm' (GT visible "
                        "cloud) or 'da3' (GT-free DA3 depth points). MUST match the checkpoint's "
                        "--geom-source, else train/eval geometry mismatch collapses the result.")
    p.add_argument("--geom-frames", type=int, default=16,
                   help="For --geom-source da3: number of frames (densely spanning the window) "
                        "used to build the DA3 geometry. The 8 token frames give sparse, holey "
                        "depth; training used 12-24 frames, so match that density here.")
    return p.parse_args()


def _resolved(path: Path) -> Path:
    return path.expanduser().resolve()


def validate_inputs(args):
    """Resolve paths and fail before model loading or output creation."""
    args.ckpt = _resolved(args.ckpt)
    args.nova_ckpt = _resolved(args.nova_ckpt)
    args.data_root = _resolved(args.data_root)
    args.sequence_dir = _resolved(args.sequence_dir) if args.sequence_dir is not None else None
    args.gt_mesh = _resolved(args.gt_mesh) if args.gt_mesh is not None else None
    args.gt_cloud = _resolved(args.gt_cloud) if args.gt_cloud is not None else None
    args.output_root = _resolved(args.output_root)
    args.decoder_ckpt = _resolved(args.decoder_ckpt) if args.decoder_ckpt is not None else None
    args.point_corrector = (_resolved(args.point_corrector)
                            if args.point_corrector is not None else None)

    errors = []

    def require_file(path: Path, label: str):
        if not path.is_file():
            errors.append(f"{label} not found: {path}")

    def require_dir(path: Path, label: str):
        if not path.is_dir():
            errors.append(f"{label} not found: {path}")

    require_file(args.ckpt, "adapter checkpoint")
    require_file(args.nova_ckpt, "NOVA3R checkpoint")
    require_file(args.nova_ckpt.parent / ".hydra" / "config.yaml",
                 "NOVA3R Hydra config")
    require_dir(args.data_root, "dataset root")
    camera_path = args.data_root / "cam_params.json"
    require_file(camera_path, "camera parameters")

    room_dir = args.sequence_dir if args.sequence_dir is not None else args.data_root / args.room
    results = room_dir / "results"
    require_dir(room_dir, "scene/sequence directory")
    require_dir(results, "RGB/depth results directory")
    require_file(room_dir / "traj.txt", "camera trajectory")

    mesh_path = args.gt_mesh if args.gt_mesh is not None else args.data_root / f"{args.room}_mesh.ply"
    require_file(mesh_path, "GT mesh")
    if args.gt_cloud is not None:
        require_file(args.gt_cloud, "GT point cloud")
    if args.decoder_ckpt is not None:
        require_file(args.decoder_ckpt, "decoder checkpoint")
    if args.point_corrector is not None:
        require_file(args.point_corrector, "point-corrector checkpoint")
    if args.output_root.exists() and not args.output_root.is_dir():
        errors.append(f"output root exists but is not a directory: {args.output_root}")

    all_fids = []
    if results.is_dir():
        rgb_paths = list(results.glob("frame*.jpg")) + list(results.glob("frame*.png"))
        if not rgb_paths:
            errors.append(f"no frame*.jpg or frame*.png RGB images found in: {results}")
        else:
            try:
                all_fids = sorted({int(path.stem.removeprefix("frame")) for path in rgb_paths})
            except ValueError:
                errors.append(f"RGB filenames must follow frameXXXXXX.jpg/png in: {results}")
            if all_fids and all_fids != list(range(len(all_fids))):
                errors.append("RGB frame IDs must be contiguous and start at 0; "
                              f"found {all_fids[0]}..{all_fids[-1]} across {len(all_fids)} frames")

    trajectory_path = room_dir / "traj.txt"
    if trajectory_path.is_file():
        try:
            pose_values = np.loadtxt(trajectory_path, dtype=np.float32)
            if pose_values.size % 16:
                errors.append(f"camera trajectory does not contain complete 4x4 poses: {trajectory_path}")
            else:
                n_poses = pose_values.size // 16
                if all_fids and n_poses != len(all_fids):
                    errors.append(f"trajectory/RGB count mismatch: {n_poses} poses, "
                                  f"{len(all_fids)} RGB frames")
        except (OSError, ValueError) as exc:
            errors.append(f"cannot parse camera trajectory {trajectory_path}: {exc}")

    if camera_path.is_file():
        try:
            with camera_path.open() as f:
                camera = json.load(f)["camera"]
            required = {"h", "w", "scale", "fx", "fy", "cx", "cy"}
            missing = sorted(required - camera.keys())
            if missing:
                errors.append(f"camera parameters missing keys {missing}: {camera_path}")
        except (OSError, ValueError, KeyError, TypeError) as exc:
            errors.append(f"cannot parse camera parameters {camera_path}: {exc}")

    if args.gtfree_tokens and args.da3pose_tokens:
        errors.append("--gtfree-tokens and --da3pose-tokens are mutually exclusive")
    if args.n_frames <= 0:
        errors.append(f"--n-frames must be positive, got {args.n_frames}")
    if args.stride <= 0:
        errors.append(f"--stride must be positive, got {args.stride}")
    if args.num_queries <= 0:
        errors.append(f"--num-queries must be positive, got {args.num_queries}")
    if args.max_windows is not None and args.max_windows <= 0:
        errors.append(f"--max-windows must be positive, got {args.max_windows}")
    if args.corrector_steps_sweep is not None and args.point_corrector is None:
        errors.append("--corrector-steps-sweep requires --point-corrector")
    if all_fids:
        span = (args.n_frames - 1) * args.stride
        if span >= len(all_fids):
            errors.append(f"not enough frames for --n-frames {args.n_frames} and "
                          f"--stride {args.stride}: found {len(all_fids)}")
        if not args.complete_target:
            missing_depth = [fid for fid in all_fids
                             if not (results / f"depth{fid:06d}.png").is_file()]
            if missing_depth:
                preview = ", ".join(map(str, missing_depth[:5]))
                suffix = "..." if len(missing_depth) > 5 else ""
                errors.append(f"missing {len(missing_depth)} depth maps in {results}; "
                              f"frame IDs: {preview}{suffix}")

    if errors:
        detail = "\n  - ".join(errors)
        raise SystemExit(f"Preflight validation failed:\n  - {detail}")

    return room_dir, results, mesh_path


def load_da3_depth_model(snapshot: Path):
    """DepthAnything3 API model for GT-free depth (separate from the token backbone)."""
    import types
    for _m in ("pycolmap",):
        sys.modules.setdefault(_m, types.ModuleType(_m))
    from depth_anything_3.api import DepthAnything3
    return DepthAnything3.from_pretrained(str(snapshot)).to(DEV).eval()


def frame_path(results: Path, fid: int) -> Path:
    """Resolve Replica/NeuralRGBD JPEGs and converted 7-Scenes PNGs."""
    for suffix in (".jpg", ".png"):
        path = results / f"frame{fid:06d}{suffix}"
        if path.exists():
            return path
    raise FileNotFoundError(f"No RGB frame for id {fid} in {results}")


def da3_window_geom(model, results, fids, poses_c2w, first_c2w, nf, K_native, m,
                    process_res=504, conf_pct=40.0):
    """DA3-predicted geometry for one stitch window, in the training pts_norm frame:
    DA3 depth -> world (GT poses) -> first-camera -> /nf*3.
    No flip at eval time. Returns [1, m, 3]."""
    from depth_anything_3.utils.export.glb import _depths_to_world_points_with_colors
    paths = [str(frame_path(results, f)) for f in fids]
    c2w = poses_c2w.numpy().astype(np.float32); w2c = np.linalg.inv(c2w)
    Kin = np.tile(K_native.numpy()[None], (len(fids), 1, 1))
    with torch.no_grad():
        p = model.inference(image=paths, extrinsics=w2c, intrinsics=Kin,
                            align_to_input_ext_scale=True, process_res=process_res)
    depth = np.asarray(p.depth); conf = None if p.conf is None else np.asarray(p.conf)
    Kp = np.asarray(p.intrinsics); H, W = depth.shape[-2:]
    cthr = np.percentile(conf, conf_pct) if conf is not None else 0.0
    da3w, _ = _depths_to_world_points_with_colors(
        depth, Kp, w2c, np.zeros((len(fids), H, W, 3), np.uint8), conf, cthr)
    da3w = da3w[np.isfinite(da3w).all(1)]
    first_cam = world_to_first_camera(torch.from_numpy(da3w).float(), first_c2w).numpy()
    da3_norm = first_cam / nf * 3.0
    # Match the cache EXACTLY: FPS (capped candidates) to 4096 for spatially-uniform
    # coverage, then uniform-subsample to m -- the distribution the adapter trained on.
    # Uniform-from-full-cloud is density-weighted and off-distribution, which collapses
    # the (geometry-sensitive) conditioned decode.
    pool = to_exact(da3_norm.astype(np.float32), 4096, use_fps=True)   # [4096, 3], FPS
    g = torch.Generator().manual_seed(0)
    idx = (torch.randperm(len(pool), generator=g)[:m] if len(pool) >= m
           else torch.randint(len(pool), (m,), generator=g))
    return torch.from_numpy(pool[idx.numpy()]).float().unsqueeze(0)


def adapter_cfg(sd):
    tt, hd = sd["target_queries"].shape
    cfg = dict(source_dim=sd["source_proj.weight"].shape[1], hidden_dim=hd,
               target_tokens=tt, target_dim=sd["out_proj.weight"].shape[0],
               depth=sum(1 for k in sd if k.endswith(".query_norm.weight")),
               num_heads=hd // 64)
    if "geom_embed.0.weight" in sd:                 # geometry-conditioned checkpoint
        cfg["geom_cond"] = True
        cfg["geom_bands"] = sd["geom_embed.0.weight"].shape[1] // 6   # in = 3*2*bands
    return cfg


@torch.no_grad()
def _run_ode(nova, ncfg, tokens, pts_norm, x_init):
    """Integrate the NOVA3R FM ODE from a given prior sample x_init -> surface."""
    solver = ODESolver(velocity_model=BatchModelWrapper(model=nova))
    step = ncfg.get("fm_step_size", 0.04)
    T = torch.linspace(0, 1, int(1 // step)).to(DEV)
    with torch.amp.autocast("cuda", enabled=False):
        sol = solver.sample(time_grid=T, x_init=x_init,
                            method=ncfg.get("fm_sampling", "euler"), step_size=step,
                            return_intermediates=False, images=torch.zeros(1, 1, 3, 1, 1, device=DEV),
                            token_mask=None, encoder_data={"tokens": tokens.to(DEV)},
                            pointmaps=pts_norm.to(DEV))
    return (sol[-1] if isinstance(sol, list) else sol)[0].cpu().float().numpy()


@torch.no_grad()
def decode(nova, ncfg, tokens, pts_norm, num_q, seed, x_init=None):
    torch.manual_seed(seed)
    if x_init is None:
        x_init = torch.rand(1, num_q, 3, device=DEV) * 2 - 1
    return _run_ode(nova, ncfg, tokens, pts_norm, x_init)


@torch.no_grad()
def decode_importance(nova, ncfg, tokens, pts_norm, num_q, extra_q, seed,
                      frac=0.35, sigma=0.03, chunk=200_000):
    """Adaptive importance-sampled decode (ODE-correct).

    Pass 1: uniform-prior decode of num_q queries -> out0 (the base cloud, keeps
            precision on the bulk surface).
    Score:  detail = inverse local density of out0 (k-NN radius). Chair legs, edges,
            and interior/concave points sit in SPARSE output regions, so a large radius
            = high detail score. This is the geometry the base uniform decode undersamples.
    Pass 2: pick the `frac` highest-detail points, take THEIR prior inits x0, and draw
            extra_q new inits as small Gaussian perturbations around them (sigma in the
            [-1,1] prior frame). Because x0 -> x1 is locally smooth, these perturbed inits
            land near the same fine-geometry regions, densifying them WITHOUT leaving the
            uniform path the field was trained on. Concatenate both passes.
    """
    torch.manual_seed(seed)
    x0 = torch.rand(1, num_q, 3, device=DEV) * 2 - 1
    out0 = _run_ode(nova, ncfg, tokens, pts_norm, x0)              # [num_q, 3]
    # inverse local density: mean distance to k nearest neighbours in the output cloud
    k = 8
    d, _ = cKDTree(out0).query(out0, k=k + 1)
    detail = d[:, 1:].mean(1)                                      # larger = sparser = finer geom
    n_sel = max(1, int(frac * num_q))
    sel = np.argpartition(-detail, n_sel - 1)[:n_sel]             # top-detail inits
    x0_sel = x0[0, sel]                                            # [n_sel, 3]
    # draw extra_q perturbed inits around the selected fine-geometry inits
    reps = torch.randint(n_sel, (extra_q,), device=DEV)
    x0_extra = (x0_sel[reps] + sigma * torch.randn(extra_q, 3, device=DEV)).clamp(-1, 1)
    outs = [out0]
    for s in range(0, extra_q, chunk):
        xi = x0_extra[s:s + chunk].unsqueeze(0)
        outs.append(_run_ode(nova, ncfg, tokens, pts_norm, xi))
    return np.concatenate(outs, 0)


def statistical_outlier_filter(pts, k=16, z=2.0):
    """Remove points whose mean k-NN distance exceeds mean + z*std over the cloud
    (classic SOR). Prunes the floating/streaky decode noise that hurts precision."""
    if len(pts) <= k:
        return np.ones(len(pts), dtype=bool)
    d, _ = cKDTree(pts).query(pts, k=k + 1)
    md = d[:, 1:].mean(1)
    return md < md.mean() + z * md.std()


def eval_cloud(pred, gt_tree, gt_pts, n=200000, thresholds=(0.01, 0.05)):
    """accuracy = pred->GT, completeness = GT->pred, chamfer, F-scores."""
    if len(pred) == 0 or len(gt_pts) == 0:
        keys = ["accuracy_m", "completeness_m", "chamfer_m", "acc_median_m", "comp_median_m"]
        out = {k: float("nan") for k in keys}
        for t in thresholds:
            out[f"F@{int(t*100)}cm"] = 0.0
            out[f"prec@{int(t*100)}cm"] = 0.0
            out[f"recall@{int(t*100)}cm"] = 0.0
        return out
    rng = np.random.default_rng(0)
    p = pred if len(pred) <= n else pred[rng.choice(len(pred), n, replace=False)]
    g = gt_pts if len(gt_pts) <= n else gt_pts[rng.choice(len(gt_pts), n, replace=False)]
    d_pg, _ = gt_tree.query(p, k=1)              # pred -> GT (accuracy)
    pred_tree = cKDTree(p)
    d_gp, _ = pred_tree.query(g, k=1)            # GT -> pred (completeness)
    acc, comp = float(d_pg.mean()), float(d_gp.mean())
    out = {"accuracy_m": acc, "completeness_m": comp, "chamfer_m": (acc + comp) / 2,
           "acc_median_m": float(np.median(d_pg)), "comp_median_m": float(np.median(d_gp))}
    for t in thresholds:
        prec = float((d_pg < t).mean()); rec = float((d_gp < t).mean())
        f = 2 * prec * rec / (prec + rec + 1e-9)
        out[f"F@{int(t*100)}cm"] = f
        out[f"prec@{int(t*100)}cm"] = prec
        out[f"recall@{int(t*100)}cm"] = rec
    return out


def main():
    args = parse_args()
    room_dir, results, mesh_path = validate_inputs(args)
    tag = args.out_tag or "default"
    out_dir = args.output_root / f"stitch_{args.room}_{tag}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[paths] repo={REPO}")
    print(f"[paths] data={args.data_root}  output={out_dir}")
    print("[load] NOVA3R"); nova, ncfg = load_nova3r_model(str(args.nova_ckpt), str(DEV))
    if args.decoder_ckpt is not None:
        sd = torch.load(args.decoder_ckpt, map_location="cpu", weights_only=True)
        missing, unexpected = nova.pts3d_head.load_state_dict(sd, strict=False)
        print(f"[decoder-ckpt] loaded FT head {args.decoder_ckpt.name} "
              f"(missing={len(missing)} unexpected={len(unexpected)})")
    nova.eval(); [pp.requires_grad_(False) for pp in nova.parameters()]
    OmegaConf.set_struct(ncfg, False)
    if args.fm_sampling is not None:  ncfg["fm_sampling"]  = args.fm_sampling
    if args.fm_step_size is not None: ncfg["fm_step_size"] = args.fm_step_size
    norm_mode = ncfg.model.params.cfg.pts3d_head.params.get("norm_mode", "none")
    print(f"       norm_mode={norm_mode}  fm_sampling={ncfg.get('fm_sampling','euler')} fm_step={ncfg.get('fm_step_size',0.04)}")

    da3_snapshot = resolve_da3_snapshot(
        args.da3_model, args.da3_revision, local_files_only=args.offline
    )
    print(f"[load] DA3 snapshot {da3_snapshot}")
    da3 = load_da3_model(da3_snapshot, DEV)

    print(f"[load] adapter ckpt {args.ckpt}")
    ck = torch.load(str(args.ckpt), map_location="cpu", weights_only=False)
    sd = ck["state_dict"]; acfg = adapter_cfg(sd); print("[load] adapter cfg", acfg)
    adapter = DA3ToNOVA3RAlignment(**acfg, drop=0.0).to(DEV); adapter.load_state_dict(sd); adapter.eval()

    corrector = None
    if args.point_corrector is not None:
        cc = torch.load(str(args.point_corrector), map_location="cpu", weights_only=False)
        ca = cc["args"]
        corrector = PointFlowCorrector(token_dim=acfg["target_dim"], hidden=ca["hidden"],
                                       depth=ca["depth"], local_knn=ca.get("local_knn", 0)).to(DEV)
        corrector.load_state_dict(cc["model"]); corrector.eval()
        print(f"[load] point corrector {args.point_corrector.name} "
              f"(hidden={ca['hidden']} depth={ca['depth']} steps={args.corrector_steps})")

    da3_depth = None
    if getattr(adapter, "geom_cond", False) and args.geom_source == "da3":
        print("[load] DA3 depth model for GT-free geometry conditioning (--geom-source da3)")
        da3_depth = load_da3_depth_model(da3_snapshot)

    da3_api = None
    if args.da3pose_tokens:
        print("[load] DA3 api model for GT-free predicted-pose tokens (--da3pose-tokens)")
        da3_api = load_da3_depth_model(da3_snapshot)  # used only for pose prediction

    # camera + geometry
    data_root = args.data_root
    with (data_root / "cam_params.json").open() as f:
        cam = json.load(f)["camera"]
    H_nat, W_nat = int(cam["h"]), int(cam["w"]); depth_scale = float(cam["scale"])
    K_native = torch.tensor([[cam["fx"], 0., cam["cx"]], [0., cam["fy"], cam["cy"]], [0., 0., 1.]], dtype=torch.float32)
    K_proc = K_native.clone(); K_proc[0] *= IMG_W / W_nat; K_proc[1] *= IMG_H / H_nat

    room = args.room
    poses_all = np.loadtxt(room_dir / "traj.txt", dtype=np.float32).reshape(-1, 4, 4)
    rgb_paths = list(results.glob("frame*.jpg")) + list(results.glob("frame*.png"))
    all_fids = sorted({int(p.stem.replace("frame", "")) for p in rgb_paths})
    n_total = len(all_fids)
    print(f"[data] {room}: {n_total} frames from {room_dir}")

    print("[data] sampling mesh surface (2M)")
    mesh = trimesh.load(str(mesh_path), force="mesh", process=False)
    mesh_pts = torch.from_numpy(trimesh.sample.sample_surface(mesh, 2_000_000)[0].astype(np.float32))

    # GT cloud: use an explicit path or the repository copy if present, else the mesh sample.
    gt_ply = (args.gt_cloud if args.gt_cloud is not None else
              REPO / f"outputs/replica/gt_pointclouds/{room}/{room}_gt_2m.ply")
    if gt_ply.exists():
        print("[data] GT cloud (prebuilt)"); gt_pts = np.asarray(trimesh.load(str(gt_ply), process=False).vertices, np.float32)
    else:
        print("[data] GT cloud (mesh surface sample)"); gt_pts = mesh_pts.numpy()
    gt_tree = cKDTree(gt_pts)
    print(f"       GT pts {len(gt_pts):,}")

    # consecutive non-overlapping windows tiling the trajectory
    S, NF = args.stride, args.n_frames
    span = (NF - 1) * S
    windows = []
    base = 0
    while base + span < n_total:
        windows.append([base + j * S for j in range(NF)])
        base += span + S  # next window starts right after the last frame of this one
    if args.max_windows:
        windows = windows[:args.max_windows]
    print(f"[plan] {len(windows)} windows, stride={S}, {NF} frames/window, span={span} frames each")

    def load_depth(fid):
        return torch.from_numpy(np.asarray(Image.open(results / f"depth{fid:06d}.png"), np.float32) / depth_scale)

    def load_rgb(fid):
        img = Image.open(frame_path(results, fid)).convert("RGB").resize((IMG_W, IMG_H), Image.BILINEAR)
        return torch.from_numpy(np.array(img, np.float32)).permute(2, 0, 1) / 255.0

    sweep_steps = ([int(x) for x in args.corrector_steps_sweep.split(",")]
                   if args.corrector_steps_sweep else None)
    if sweep_steps is not None and corrector is None:
        raise SystemExit("--corrector-steps-sweep requires --point-corrector")
    sweep_world = {s: [] for s in sweep_steps} if sweep_steps is not None else None

    pred_world_all, oracle_world_all, input_world_all = [], [], []
    per_window = []

    for wi, fids in enumerate(windows):
        poses_c2w = torch.from_numpy(np.stack([poses_all[f] for f in fids])).float()
        first_c2w = poses_c2w[0]

        # pool in first-camera frame: visible (occlusion-culled) or complete (frustum, amodal)
        pool_list = []
        for i, fid in enumerate(fids):
            if args.complete_target:
                proj = project_world_points(points_world=mesh_pts,
                    world_to_camera=torch.linalg.inv(poses_c2w[i]), intrinsics=K_native,
                    image_hw=(H_nat, W_nat))
                pw = mesh_pts[proj["inside"]]
            else:
                pw = crop_visible_world_points(points_world=mesh_pts, camera_to_world=poses_c2w[i],
                    intrinsics=K_native, depth=load_depth(fid), depth_tolerance=0.05)["points_world"]
            pool_list.append(world_to_first_camera(pw, first_c2w))
        pool = torch.cat(pool_list, dim=0)

        # sample K, normalize via NOVA3R encoder -> pts_norm, metric norm_factor, encoder tokens z
        gen = torch.Generator().manual_seed(0)
        if pool.shape[0] >= K_SAMPLE:
            idx = torch.randperm(pool.shape[0], generator=gen)[:K_SAMPLE]
        else:
            pad = torch.randint(pool.shape[0], (K_SAMPLE - pool.shape[0],), generator=gen)
            idx = torch.cat([torch.arange(pool.shape[0]), pad])
        pts = pool[idx].unsqueeze(0).to(DEV).float()
        valid = torch.ones(pts.shape[:2], dtype=torch.bool, device=DEV)
        with torch.no_grad():
            pts_norm, _ = normalize_input(pts, valid, pts, valid, mode=norm_mode)
            z_enc = nova._encode(pointmaps=pts_norm, test=True)["tokens"].float()
        nf = float(pts.cpu()[0].norm(dim=-1).median().clamp(0.01, 100.0))

        # l13 DA3 tokens (exactly as training cache)
        images = torch.stack([load_rgb(f) for f in fids])
        intrinsics = torch.from_numpy(np.tile(K_proc.numpy()[None], (NF, 1, 1))).float()
        if args.da3pose_tokens:
            # GT-FREE: DA3 predicts poses, fed to cam_enc (matches --pose-source da3pred training)
            wp = [str(frame_path(results, f)) for f in fids]
            pp = da3_api.inference(image=wp, extrinsics=None, intrinsics=None, process_res=IMG_H)
            ex = np.asarray(pp.extrinsics)
            if ex.shape[-2:] == (3, 4):
                ex = np.concatenate([ex, np.tile(np.array([0, 0, 0, 1], np.float32), (len(fids), 1, 1))], axis=1)
            c2w_pred = torch.from_numpy(np.linalg.inv(ex)).float()
            da3_tokens = extract_da3_tokens(da3, images, c2w_pred, K_proc[None].repeat(NF, 1, 1),
                                            layer_idx=DA3_LAYERS, max_tokens=DA3_MAX_TOKENS, device=DEV,
                                            use_poses=True)
        else:
            da3_tokens = extract_da3_tokens(da3, images, poses_c2w, intrinsics,
                                            layer_idx=DA3_LAYERS, max_tokens=DA3_MAX_TOKENS, device=DEV,
                                            use_poses=not args.gtfree_tokens)

        with torch.no_grad():
            geom = None
            if getattr(adapter, "geom_cond", False):
                # feed the SAME geometry source the adapter trained on, in the pts_norm frame.
                if da3_depth is not None:                          # --geom-source da3 (GT-free)
                    # Build DA3 geometry from a DENSER frame set spanning the window (the 8
                    # token frames give holey depth; training used 12-24 frames).
                    gf = np.unique(np.linspace(fids[0], fids[-1], args.geom_frames).round().astype(int))
                    gf = [f for f in gf if f in set(all_fids)]
                    gposes = torch.from_numpy(np.stack([poses_all[f] for f in gf])).float()
                    geom = da3_window_geom(da3_depth, results, gf, gposes, first_c2w,
                                           nf, K_native, args.geom_points).to(DEV)
                else:                                              # --geom-source pts_norm
                    pn = pts_norm[0]                               # [K, 3]
                    m = min(args.geom_points, pn.shape[0])
                    gi = torch.randperm(pn.shape[0], generator=torch.Generator().manual_seed(0))[:m]
                    geom = pn[gi].unsqueeze(0).to(DEV)
            z_pred = adapter(da3_tokens.to(DEV), geom_xyz=geom).cpu()
        if args.importance:
            pred_n = decode_importance(nova, ncfg, z_pred, pts_norm.cpu(), args.num_queries,
                                       args.imp_extra, args.seed, args.imp_frac, args.imp_sigma)
            orac_n = decode_importance(nova, ncfg, z_enc.cpu(), pts_norm.cpu(), args.num_queries,
                                       args.imp_extra, args.seed, args.imp_frac, args.imp_sigma)
        else:
            pred_n = decode(nova, ncfg, z_pred, pts_norm.cpu(), args.num_queries, args.seed)
            orac_n = decode(nova, ncfg, z_enc.cpu(), pts_norm.cpu(), args.num_queries, args.seed)

        R, t = first_c2w[:3, :3].numpy(), first_c2w[:3, 3].numpy()
        to_world = lambda pn: (R @ (pn / 3.0 * nf).T).T + t

        # Optional Stage-2 point-space correction: snap the predicted cloud toward the real
        # surface with the frozen flow corrector, conditioned only on z_pred tokens (no GT).
        if corrector is not None and sweep_steps is not None:
            # decode-once sweep: apply the corrector at each step count to the SAME decoded cloud
            x0 = torch.from_numpy(pred_n).float().unsqueeze(0).to(DEV)
            for s in sweep_steps:
                cn = (pred_n if s == 0 else
                      integrate_point_corrector(corrector, x0, z_pred.to(DEV), s)[0].cpu().numpy())
                sweep_world[s].append(to_world(cn))
        elif corrector is not None:
            x0 = torch.from_numpy(pred_n).float().unsqueeze(0).to(DEV)
            corr = integrate_point_corrector(
                corrector, x0, z_pred.to(DEV), args.corrector_steps
            )
            pred_n = corr[0].cpu().numpy()

        pred_w, orac_w = to_world(pred_n), to_world(orac_n)
        input_w = (R @ (pts_norm[0].cpu().numpy() / 3.0 * nf).T).T + t

        pred_world_all.append(pred_w); oracle_world_all.append(orac_w); input_world_all.append(input_w)

        d_pred = gt_tree.query(pred_w, k=1)[0]
        d_orac = gt_tree.query(orac_w, k=1)[0]
        per_window.append((fids[0], fids[-1], nf, float(d_pred.mean()), float(d_orac.mean())))
        print(f"  win {wi:02d} frames {fids[0]:4d}-{fids[-1]:4d}  pool={pool.shape[0]:>7,}  "
              f"nf={nf:.3f}m  da3={tuple(da3_tokens.shape)}  "
              f"pred->GT={d_pred.mean():.4f}m  oracle->GT={d_orac.mean():.4f}m", flush=True)

    # ---- corrector-steps sweep: eval FURN F@2 per step count, then return ----
    if sweep_steps is not None:
        gt_pts_z = gt_pts[:, 2]
        floor = float(np.percentile(gt_pts_z, 1))
        xy_lo = np.percentile(gt_pts[:, :2], 1, axis=0)
        xy_hi = np.percentile(gt_pts[:, :2], 99, axis=0)
        def fmask(p, margin=0.4, zlo=0.15, zhi=1.1):
            zz = p[:, 2]
            return ((zz > floor + zlo) & (zz < floor + zhi)
                    & (p[:, 0] > xy_lo[0] + margin) & (p[:, 0] < xy_hi[0] - margin)
                    & (p[:, 1] > xy_lo[1] + margin) & (p[:, 1] < xy_hi[1] - margin))
        gt_furn = gt_pts[fmask(gt_pts)]
        gt_furn_tree = cKDTree(gt_furn)
        print(f"\n[sweep] corrector-steps FURN F@2 (0 = uncorrected), {len(gt_furn):,} GT furn pts:")
        sweep_out = {}
        for s in sweep_steps:
            cloud = np.concatenate(sweep_world[s], 0)
            m = eval_cloud(cloud[fmask(cloud)], gt_furn_tree, gt_furn, thresholds=(0.02, 0.05))
            sweep_out[s] = m
            print(f"   steps={s:3d}  FURN F@2={m['F@2cm']:.4f}  prec={m['prec@2cm']:.3f}  "
                  f"rec={m['recall@2cm']:.3f}", flush=True)
        import json as _json
        (out_dir / "sweep_metrics.json").write_text(_json.dumps(
            {str(s): sweep_out[s] for s in sweep_steps}, indent=2))
        print(f"[sweep] -> {out_dir/'sweep_metrics.json'}")
        return

    pred_world = np.concatenate(pred_world_all, 0)
    oracle_world = np.concatenate(oracle_world_all, 0)
    input_world = np.concatenate(input_world_all, 0)

    if args.filter_sor:
        kp = statistical_outlier_filter(pred_world, args.sor_k, args.sor_z)
        ko = statistical_outlier_filter(oracle_world, args.sor_k, args.sor_z)
        print(f"\n[filter] SOR (k={args.sor_k} z={args.sor_z}): "
              f"pred {len(pred_world):,}->{int(kp.sum()):,} ({100*kp.mean():.1f}% kept)  "
              f"oracle {len(oracle_world):,}->{int(ko.sum()):,} ({100*ko.mean():.1f}% kept)")
        pred_world, oracle_world = pred_world[kp], oracle_world[ko]

    print("\n[eval] stitched PRED vs GT")
    m_pred = eval_cloud(pred_world, gt_tree, gt_pts)
    print("[eval] stitched ORACLE (NOVA3R AE ceiling) vs GT")
    m_orac = eval_cloud(oracle_world, gt_tree, gt_pts)
    print("[eval] stitched INPUT pool vs GT (geometry sanity)")
    m_in = eval_cloud(input_world, gt_tree, gt_pts)

    # ---- furniture-region detail metric ----
    # Objects standing on the floor (table/chairs/corner clutter) live in an
    # interior height band, away from floor/ceiling and away from the wall shell.
    # Bounds are derived once from GT so GT and pred share identical thresholds.
    z = gt_pts[:, 2]
    floor = float(np.percentile(z, 1))
    xy_lo = np.percentile(gt_pts[:, :2], 1, axis=0)
    xy_hi = np.percentile(gt_pts[:, :2], 99, axis=0)
    def fmask(p, margin=0.4, zlo=0.15, zhi=1.1):
        zz = p[:, 2]
        return ((zz > floor + zlo) & (zz < floor + zhi)
                & (p[:, 0] > xy_lo[0] + margin) & (p[:, 0] < xy_hi[0] - margin)
                & (p[:, 1] > xy_lo[1] + margin) & (p[:, 1] < xy_hi[1] - margin))
    gt_furn = gt_pts[fmask(gt_pts)]
    print(f"\n[eval] furniture region: {len(gt_furn):,} GT pts "
          f"(z∈[{floor+0.15:.2f},{floor+1.1:.2f}] + interior crop). Chairs/table/corner objects.")
    gt_furn_tree = cKDTree(gt_furn)
    fm_pred = eval_cloud(pred_world[fmask(pred_world)], gt_furn_tree, gt_furn, thresholds=(0.02, 0.05))
    fm_orac = eval_cloud(oracle_world[fmask(oracle_world)], gt_furn_tree, gt_furn, thresholds=(0.02, 0.05))

    def show(name, m):
        print(f"\n=== {name} ===")
        for k, v in m.items():
            print(f"   {k:18s} {v:.4f}")
    show("PRED (adapter)", m_pred)
    show("ORACLE (encoder ceiling)", m_orac)
    show("INPUT (visible pool)", m_in)
    show("PRED  furniture-region", fm_pred)
    show("ORACLE furniture-region", fm_orac)

    trimesh.PointCloud(pred_world).export(out_dir / f"{room}_pred_stitched.ply")
    trimesh.PointCloud(oracle_world).export(out_dir / f"{room}_oracle_stitched.ply")
    trimesh.PointCloud(input_world).export(out_dir / f"{room}_input_stitched.ply")

    # per-window arrays for the accumulation animation
    np.savez_compressed(
        out_dir / "per_window.npz",
        pred=np.stack(pred_world_all).astype(np.float32),       # [W, Q, 3]
        oracle=np.stack(oracle_world_all).astype(np.float32),
        input=np.stack(input_world_all).astype(np.float32),
        cam0=np.stack([poses_all[fids[0]] for fids in windows]).astype(np.float32),  # [W,4,4]
        f0=np.array([fids[0] for fids in windows]),
        f1=np.array([fids[-1] for fids in windows]),
    )
    print(f"\n[out] PLYs -> {out_dir}")
    print(f"[out] pred pts {len(pred_world):,}  oracle {len(oracle_world):,}  input {len(input_world):,}")

    import json as _json
    (out_dir / "metrics.json").write_text(_json.dumps(
        {"ckpt": str(args.ckpt), "pred": m_pred, "oracle": m_orac, "input": m_in,
         "pred_furniture": fm_pred, "oracle_furniture": fm_orac,
         "n_windows": len(windows), "stride": S, "n_frames": NF, "num_queries": args.num_queries,
         "per_window": [{"f0": a, "f1": b, "nf": c, "pred_to_gt": d, "oracle_to_gt": e}
                        for (a, b, c, d, e) in per_window]}, indent=2))
    print(f"[out] metrics -> {out_dir/'metrics.json'}")


if __name__ == "__main__":
    main()
