#!/usr/bin/env python3
"""
Test the canonical camera pose hypothesis.

Decodes three things for one "val" window and displays them in camera space
(normalised, BEFORE any c2w transform) and world space:

  1. Input GT pts_norm          — what the encoder saw
  2. GT z_star decoded          — ground truth target; should be flat
  3. Adapter z_pred decoded     — what the adapter outputs; curved in practice?

If the curvature is present in camera space for (3) but not (1)/(2), the
distortion is baked into z_pred itself — a rigid c2w transform cannot cause it.
This rules out a simple camera-frame c2w mismatch and points to the adapter
predicting geometrically wrong tokens.

Cross-frame test: for each other cached window ("training window"), the script
also decodes that window's GT z_star and then applies the VAL c2w instead of
the training c2w. If this "wrong-c2w" version looks like the adapter pred, the
adapter is memorising a training window's z_star.

Outputs:
  - Printed Chamfer distances (pred vs GT pts, gt_decoded vs GT pts,
    pred vs each training decoded cloud)
  - Rerun .rrd file with camera-space and world-space views

Usage:
  ! python experiments/test_pose_hypothesis.py \\
      --checkpoint outputs/consecutive_windows/hungarian_online_v20_420_820_1220_1620_best.pt \\
      --data-root scripts/data/windows_s20_room0 \\
      --val-start 0 \\
      --rrd-out outputs/pose_hypothesis.rrd
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

REPO_ROOT   = Path(__file__).resolve().parents[1]
NOVA3R_ROOT = REPO_ROOT / "nova3r"
NOVA3R_3P   = NOVA3R_ROOT / "third_party"
DA3_SRC     = REPO_ROOT / "da3" / "src"
OVERFIT_SRC = REPO_ROOT / "experiments" / "overfit_8frames"

for _p in [str(NOVA3R_3P), str(NOVA3R_ROOT), str(DA3_SRC), str(OVERFIT_SRC)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from demo_nova3r import load_model as load_nova3r_model                    # noqa: E402
from nova3r.models.model_wrapper import BatchModelWrapper                  # noqa: E402
from nova3r.flow_matching.solver import ODESolver                          # noqa: E402
from nova3r.inference import amp_dtype_mapping                             # noqa: E402
from vc3r.alignment import DA3ToNOVA3RAlignment  # noqa: E402

CFG_PATH = OVERFIT_SRC / "config.yaml"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--data-root",  type=Path,
                   default=REPO_ROOT / "scripts" / "data" / "windows_s20_room0")
    p.add_argument("--val-start",  type=int, default=0)
    p.add_argument("--rrd-out",    type=Path,
                   default=REPO_ROOT / "outputs" / "pose_hypothesis.rrd")
    p.add_argument("--num-decode", type=int, default=4096)
    p.add_argument("--seed",       type=int, default=42)
    p.add_argument("--device",     default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def load_window(data_root: Path, start: int):
    win_dir = data_root / f"start_{start:04d}"
    meta      = torch.load(win_dir / "meta.pt",             map_location="cpu", weights_only=False)
    da3       = torch.load(win_dir / "da3_tokens.pt",       map_location="cpu", weights_only=True)
    z_star    = torch.load(win_dir / "z_star_consensus.pt", map_location="cpu", weights_only=True)
    pts_norm  = torch.load(win_dir / "pts_norm.pt",         map_location="cpu", weights_only=True)
    return meta, da3, z_star, pts_norm


@torch.no_grad()
def decode(nova_model, nova_cfg, tokens: torch.Tensor, pts_norm: torch.Tensor,
           device: torch.device, num_decode: int, seed: int) -> np.ndarray:
    torch.manual_seed(seed)
    encoder_data = {"tokens": tokens.to(device)}
    images  = torch.zeros(1, 1, 3, 1, 1, device=device)
    x_init  = torch.rand(1, num_decode, 3, device=device) * 2 - 1
    wrapper = BatchModelWrapper(model=nova_model)
    solver  = ODESolver(velocity_model=wrapper)
    step_sz = nova_cfg.get("fm_step_size", 0.04)
    method  = nova_cfg.get("fm_sampling", "euler")
    amp_dt  = amp_dtype_mapping.get(nova_cfg.get("amp_dtype", "bf16"), torch.float32)
    T_grid  = torch.linspace(0, 1, int(1 // step_sz)).to(device)
    use_amp = device.type != "cpu"
    with torch.cuda.amp.autocast(enabled=use_amp, dtype=amp_dt):
        sol = solver.sample(
            time_grid=T_grid, x_init=x_init, method=method,
            step_size=step_sz, return_intermediates=False,
            images=images, token_mask=None,
            encoder_data=encoder_data, pointmaps=pts_norm.to(device),
        )
    return (sol[-1] if isinstance(sol, list) else sol)[0].cpu().float().numpy()


def to_world(cam_pts: np.ndarray, norm_factor: float, c2w: np.ndarray) -> np.ndarray:
    pts = cam_pts / 3.0 * norm_factor
    ones = np.ones((len(pts), 1), dtype=np.float32)
    return (c2w @ np.hstack([pts, ones]).T).T[:, :3].astype(np.float32)


def chamfer(a: np.ndarray, b: np.ndarray, n: int = 8192) -> float:
    rng = np.random.default_rng(0)
    if len(a) > n:
        a = a[rng.choice(len(a), n, replace=False)]
    if len(b) > n:
        b = b[rng.choice(len(b), n, replace=False)]
    a_t, b_t = torch.from_numpy(a).float(), torch.from_numpy(b).float()
    d = torch.cdist(a_t, b_t)
    return float((d.min(1).values.mean() + d.min(0).values.mean()) / 2)


def camera_yaw_deg(c2w: np.ndarray) -> float:
    R = c2w[:3, :3]
    return float(np.arctan2(R[1, 0], R[0, 0]) * 180 / np.pi)


def sub(pts: np.ndarray, n: int = 50_000) -> np.ndarray:
    if len(pts) <= n:
        return pts
    return pts[np.random.default_rng(1).choice(len(pts), n, replace=False)]


def shaded(pts: np.ndarray, base: np.ndarray) -> np.ndarray:
    axis = 1  # Y axis
    lo, hi = pts[:, axis].min(), pts[:, axis].max()
    t = np.clip((pts[:, axis] - lo) / (hi - lo + 1e-8), 0, 1)
    shade = 0.4 + 0.6 * t
    return np.clip(base[None].astype(np.float32) * shade[:, None], 0, 255).astype(np.uint8)


def main() -> None:
    args   = p = parse_args()
    device = torch.device(args.device)
    cfg    = OmegaConf.load(CFG_PATH)
    OmegaConf.set_struct(cfg, False)

    # ── load adapter ──────────────────────────────────────────────────────────
    ckpt_data = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    adapter = DA3ToNOVA3RAlignment(
        source_dim=int(cfg.source_dim), hidden_dim=int(cfg.hidden_dim),
        target_tokens=int(cfg.target_tokens), target_dim=int(cfg.target_dim),
        depth=int(cfg.depth), num_heads=int(cfg.num_heads), drop=0.0,
    ).to(device)
    adapter.load_state_dict(ckpt_data["state_dict"])
    adapter.eval()
    print(f"[ckpt] {args.checkpoint.name}  step={ckpt_data.get('step')}  "
          f"val_loss={ckpt_data.get('val_loss'):.5f}")

    # ── load NOVA3R ───────────────────────────────────────────────────────────
    print("[model] Loading NOVA3R …")
    nova_ckpt = str(NOVA3R_ROOT / "checkpoints" / "scene_ae" / "checkpoint-last.pth")
    nova_model, nova_cfg = load_nova3r_model(nova_ckpt, str(device))
    nova_model.eval()
    for p in nova_model.parameters():
        p.requires_grad_(False)
    OmegaConf.set_struct(nova_cfg, False)

    # ── load val window ───────────────────────────────────────────────────────
    val_meta, val_da3, val_z, val_pts_norm = load_window(args.data_root, args.val_start)
    val_nf  = float(val_meta["norm_factor"])
    val_c2w = val_meta["first_c2w"].numpy()
    print(f"\n[val] start={args.val_start}  frames={val_meta['frame_ids'][:2]}…  "
          f"norm_factor={val_nf:.3f}  yaw={camera_yaw_deg(val_c2w):.1f}°")

    # ── adapter prediction ────────────────────────────────────────────────────
    with torch.no_grad():
        z_pred = adapter(val_da3.to(device))    # (1, 768, 128)

    pred_norm = z_pred[0].norm(dim=-1).mean().item()
    gt_norm   = val_z[0].norm(dim=-1).mean().item()
    print(f"\n[tokens] pred token-norm mean={pred_norm:.4f}  gt token-norm mean={gt_norm:.4f}  "
          f"ratio={pred_norm/gt_norm:.3f}")

    # ── decode val: GT z_star ─────────────────────────────────────────────────
    print("\n[decode] GT z_star …")
    gt_dec_cam = decode(nova_model, nova_cfg, val_z, val_pts_norm, device, args.num_decode, args.seed)
    gt_dec_world = to_world(gt_dec_cam, val_nf, val_c2w)

    # ── decode val: adapter prediction ───────────────────────────────────────
    print("[decode] adapter z_pred …")
    pred_dec_cam = decode(nova_model, nova_cfg, z_pred, val_pts_norm, device, args.num_decode, args.seed)
    pred_dec_world = to_world(pred_dec_cam, val_nf, val_c2w)

    # ── input pts in world space ──────────────────────────────────────────────
    input_cam_np = val_pts_norm[0].numpy()            # (K, 3) normalised
    input_world  = to_world(input_cam_np, val_nf, val_c2w)

    # ── Chamfer in camera space (planarity lives here) ────────────────────────
    print("\n[chamfer] Camera space (normalised) vs input pts_norm:")
    cd_gt_cam   = chamfer(gt_dec_cam,   input_cam_np)
    cd_pred_cam = chamfer(pred_dec_cam, input_cam_np)
    print(f"  GT decoded   vs input: {cd_gt_cam:.4f}")
    print(f"  Pred decoded vs input: {cd_pred_cam:.4f}  (ratio {cd_pred_cam/cd_gt_cam:.2f}×)")

    # ── load training windows ─────────────────────────────────────────────────
    all_starts = sorted(
        int(d.name.split("_")[1])
        for d in args.data_root.glob("start_*")
        if (d / "meta.pt").exists()
    )
    train_starts = [s for s in all_starts if s != args.val_start]
    print(f"\n[cross-frame] {len(train_starts)} training windows")

    train_results = []
    for ts in train_starts:
        tr_meta, _, tr_z, tr_pts_norm = load_window(args.data_root, ts)
        tr_nf  = float(tr_meta["norm_factor"])
        tr_c2w = tr_meta["first_c2w"].numpy()
        yaw    = camera_yaw_deg(tr_c2w)

        print(f"  start={ts:4d}  yaw={yaw:.1f}°  norm_factor={tr_nf:.3f} … decoding")

        # Decode training z_star with its OWN c2w → correct world geometry
        tr_dec_cam   = decode(nova_model, nova_cfg, tr_z, tr_pts_norm, device, args.num_decode, args.seed)
        tr_dec_world = to_world(tr_dec_cam, tr_nf, tr_c2w)

        # Decode training z_star but apply VAL c2w → "wrong frame" test
        # (same pts_norm scale trick: use val norm_factor to match scale)
        tr_dec_wrong_world = to_world(tr_dec_cam, val_nf, val_c2w)

        # How similar is the adapter pred (in cam space) to this training decode?
        cd_pred_vs_tr_cam   = chamfer(pred_dec_cam, tr_dec_cam)
        cd_pred_vs_tr_wrong = chamfer(pred_dec_world, tr_dec_wrong_world)

        print(f"    pred cam vs train cam:        {cd_pred_vs_tr_cam:.4f}")
        print(f"    pred world vs train(wrong c2w):{cd_pred_vs_tr_wrong:.4f}")

        train_results.append({
            "start": ts, "yaw": yaw,
            "dec_cam": tr_dec_cam, "dec_world": tr_dec_world,
            "dec_wrong_world": tr_dec_wrong_world,
            "cd_pred_vs_tr_cam": cd_pred_vs_tr_cam,
            "cd_pred_vs_tr_wrong": cd_pred_vs_tr_wrong,
            "c2w": tr_c2w,
        })

    # ── summary ───────────────────────────────────────────────────────────────
    best_cam   = min(train_results, key=lambda r: r["cd_pred_vs_tr_cam"])
    best_wrong = min(train_results, key=lambda r: r["cd_pred_vs_tr_wrong"])
    print("\n[summary]")
    print(f"  Pred most similar to training window (cam space):        "
          f"start={best_cam['start']}  yaw={best_cam['yaw']:.1f}°  "
          f"cd={best_cam['cd_pred_vs_tr_cam']:.4f}")
    print(f"  Pred most similar to training window (wrong c2w world):  "
          f"start={best_wrong['start']}  yaw={best_wrong['yaw']:.1f}°  "
          f"cd={best_wrong['cd_pred_vs_tr_wrong']:.4f}")
    print(f"  Val window yaw: {camera_yaw_deg(val_c2w):.1f}°")
    print(f"\n  → if best_cam start ≠ val_start and cd << cd_gt_cam, adapter is "
          f"imitating a training window's z_star in its own camera frame.")

    # ── Rerun visualisation ───────────────────────────────────────────────────
    args.rrd_out.parent.mkdir(parents=True, exist_ok=True)
    try:
        import rerun as rr
    except ImportError:
        print("\n[rerun] not installed — skipping visualisation")
        return

    rr.init("pose_hypothesis", spawn=False)
    rr.save(str(args.rrd_out))
    rr.log("world", rr.ViewCoordinates.RDF, static=True)

    green  = np.array([45,  205,  90], dtype=np.uint8)
    blue   = np.array([35,  115, 255], dtype=np.uint8)
    red    = np.array([230,  45,  55], dtype=np.uint8)
    gray   = np.array([160, 160, 160], dtype=np.uint8)
    orange = np.array([255, 140,   0], dtype=np.uint8)

    # ── camera space (normalised units) ──────────────────────────────────────
    rr.set_time("space", sequence=0)
    inp_s   = sub(input_cam_np)
    gt_s    = sub(gt_dec_cam)
    pred_s  = sub(pred_dec_cam)
    rr.log("cam/input_pts",   rr.Points3D(inp_s,  colors=shaded(inp_s,  green), radii=0.008))
    rr.log("cam/gt_decoded",  rr.Points3D(gt_s,   colors=shaded(gt_s,   blue),  radii=0.010))
    rr.log("cam/pred_decoded",rr.Points3D(pred_s, colors=shaded(pred_s, red),   radii=0.010))

    # ── world space ───────────────────────────────────────────────────────────
    rr.set_time("space", sequence=1)
    inp_w  = sub(input_world)
    gtw    = sub(gt_dec_world)
    predw  = sub(pred_dec_world)
    rr.log("world/input_pts",   rr.Points3D(inp_w,  colors=shaded(inp_w,  green), radii=0.008))
    rr.log("world/gt_decoded",  rr.Points3D(gtw,    colors=shaded(gtw,    blue),  radii=0.010))
    rr.log("world/pred_decoded",rr.Points3D(predw,  colors=shaded(predw,  red),   radii=0.010))
    cameras_world = np.array([tr["c2w"][:3, 3] for tr in train_results] + [val_c2w[:3, 3]])
    rr.log("world/cameras",     rr.Points3D(cameras_world, colors=[[255,255,0]]*len(cameras_world), radii=0.06))

    # ── training windows: correct c2w (world) ────────────────────────────────
    rr.set_time("space", sequence=2)
    for tr in train_results:
        s_tr = sub(tr["dec_world"])
        rr.log(f"world/train_correct/start_{tr['start']:04d}",
               rr.Points3D(s_tr, colors=shaded(s_tr, gray), radii=0.008))

    # ── training windows: wrong c2w (val c2w applied to training cloud) ───────
    rr.set_time("space", sequence=3)
    for tr in train_results:
        s_wr = sub(tr["dec_wrong_world"])
        rr.log(f"world/train_wrong_c2w/start_{tr['start']:04d}",
               rr.Points3D(s_wr, colors=shaded(s_wr, orange), radii=0.008))

    rr.log("legend", rr.TextDocument(
        "# Pose hypothesis test\n\n"
        "**Camera space** (space=0): curvature present in camera space "
        "→ distortion is in z_pred itself, not from c2w\n\n"
        "**World space** (space=1): pred decoded with correct val c2w\n\n"
        "**Training correct c2w** (space=2, gray): each training window's "
        "z_star decoded with its own c2w — should all look flat\n\n"
        "**Training wrong c2w** (space=3, orange): each training z_star "
        "decoded with the VAL c2w — if this looks like pred, adapter "
        "is imitating that training window's z_star\n\n"
        f"Val start={args.val_start}  yaw={camera_yaw_deg(val_c2w):.1f}°\n\n"
        f"Pred token-norm / GT token-norm = {pred_norm/gt_norm:.3f}\n\n"
        f"CD(pred_cam, input): {cd_pred_cam:.4f}   CD(gt_cam, input): {cd_gt_cam:.4f}",
        media_type=rr.MediaType.MARKDOWN,
    ))

    print(f"\n[rerun] saved → {args.rrd_out}")


if __name__ == "__main__":
    main()
