#!/usr/bin/env python3
"""
Train the DA3→NOVA3R adapter with online Hungarian matching loss.

At each training step, the optimal bijection between predicted tokens and GT
consensus tokens is computed per sample via Hungarian matching. MSE is then
computed on matched pairs. Gradients flow through the MSE only — the
assignment is detached.

This removes the static reference-window dependency of pre-alignment: no window
ordering is chosen upfront, and the matching updates as the adapter improves.

Usage:
  ! python scripts/train_hungarian.py
  ! python scripts/train_hungarian.py --val-start 56 --n-left 7 --n-right 7
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from scipy.optimize import linear_sum_assignment

REPO_ROOT   = Path(__file__).resolve().parents[1]
NOVA3R_ROOT = REPO_ROOT / "nova3r"
NOVA3R_3P   = NOVA3R_ROOT / "third_party"
DA3_SRC     = REPO_ROOT / "da3" / "src"
OVERFIT_SRC = REPO_ROOT / "experiments" / "overfit_8frames"

for _p in [str(NOVA3R_3P), str(NOVA3R_ROOT), str(DA3_SRC), str(OVERFIT_SRC)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from demo_nova3r import load_model as load_nova3r_model                        # noqa: E402
from nova3r.models.model_wrapper import BatchModelWrapper                      # noqa: E402
from nova3r.flow_matching.solver import ODESolver                              # noqa: E402
from nova3r.inference import amp_dtype_mapping                                 # noqa: E402
from vc3r.alignment import DA3ToNOVA3RAlignment      # noqa: E402

DATA_ROOT = REPO_ROOT / "scripts" / "data" / "windows"
CFG_PATH  = OVERFIT_SRC / "config.yaml"
N_FRAMES  = 8


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--val-starts",  type=int,  nargs="+", default=None,
                   help="Validation window start frames (same starts used in every room). "
                        "Defaults to 56 unless --val-root is supplied, in which case all "
                        "cached start_* windows in each validation root are used.")
    p.add_argument("--train-starts", type=int, nargs="*", default=None,
                   help="Explicit training window starts for single-root mode")
    p.add_argument("--rooms", type=str, nargs="*", default=None,
                   help="Multi-room mode: 'root:s0,s1,...' per room, or bare 'root' to use "
                        "all cached start_* windows. Overrides --data-root/--train-starts.")
    p.add_argument("--frame-stride", type=int, default=1,
                   help="Frame stride used when windows were cached (controls disjointness check)")
    p.add_argument("--steps",       type=int,   default=2000)
    p.add_argument("--batch-size",  type=int,   default=None,
                   help="Training batch size in windows; defaults to full batch")
    p.add_argument("--lr",          type=float, default=1e-3)
    p.add_argument("--log-every",   type=int,   default=200)
    p.add_argument("--num-queries", type=int,   default=8192)
    p.add_argument("--seed",        type=int,   default=42)
    p.add_argument("--rrd-out",     type=Path,  default=None)
    p.add_argument("--shade-axis",  choices=["x", "y", "z"], default="y",
                   help="World axis used to shade each base visualization color")
    p.add_argument("--rot90-axis",  choices=["none", "x", "y", "z", "-x", "-y", "-z"],
                   default="none",
                   help="Rotate visualized points/cameras by 90 degrees around this world axis")
    p.add_argument("--load-ckpt",   type=Path,  default=None,
                   help="Load checkpoint and skip training (decode + Rerun only)")
    p.add_argument("--init-ckpt",   type=Path,  default=None,
                   help="Initialise adapter weights from checkpoint before training (warm start)")
    p.add_argument("--data-root",   type=Path,  default=DATA_ROOT,
                   help="Cache root for single-room mode")
    p.add_argument("--val-root",    type=Path,  nargs="*", default=None,
                   help="Held-out scene root(s) for validation. If --val-starts is omitted, "
                        "all cached start_* windows from every validation root are loaded; "
                        "default loads one val set per training room")
    p.add_argument("--val-rooms", type=str, nargs="*", default=None,
                   help="Validation roots as 'root:s0,s1,...'. Overrides --val-root/--val-starts.")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def discover_train_starts(data_root: Path, val_starts: list[int], frame_stride: int) -> list[int]:
    """All cached window starts that don't overlap any val window's frame range."""
    window_frames = N_FRAMES * frame_stride  # frames covered by one window
    val_ranges = [(v, v + window_frames - 1) for v in val_starts]
    starts = []
    for d in sorted(data_root.glob("start_*")):
        s = int(d.name.split("_")[1])
        t_lo, t_hi = s, s + window_frames - 1
        if not any(t_lo <= v_hi and t_hi >= v_lo for v_lo, v_hi in val_ranges):
            starts.append(s)
    return starts


def discover_cached_starts(data_root: Path) -> list[int]:
    """All cached window starts under one scene root."""
    starts = []
    for d in sorted(data_root.glob("start_*")):
        starts.append(int(d.name.split("_")[1]))
    if not starts:
        raise FileNotFoundError(f"No cached start_* windows found under {data_root}.")
    return starts


def load_window(start: int, data_root: Path):
    win_dir = data_root / f"start_{start:04d}"
    if not win_dir.exists():
        raise FileNotFoundError(f"Window start={start} not cached under {data_root}.")
    meta      = torch.load(win_dir / "meta.pt",             map_location="cpu", weights_only=False)
    da3       = torch.load(win_dir / "da3_tokens.pt",       map_location="cpu", weights_only=True)
    consensus = torch.load(win_dir / "z_star_consensus.pt", map_location="cpu", weights_only=True)
    pts_norm  = torch.load(win_dir / "pts_norm.pt",         map_location="cpu", weights_only=True)
    return meta, da3, consensus, pts_norm


def hungarian_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    pred, target: (N, 768, 128)
    Per-sample optimal bijection via Hungarian matching; MSE on matched pairs.
    Gradients flow through the MSE only — the assignment is detached.
    """
    losses = []
    for i in range(pred.shape[0]):
        cost = torch.cdist(
            pred[i].detach().unsqueeze(0),
            target[i].unsqueeze(0),
        )[0].cpu().numpy()                      # (768, 768)
        _, col_ind = linear_sum_assignment(cost)
        losses.append(F.mse_loss(pred[i], target[i][col_ind]))
    return torch.stack(losses).mean()


@torch.no_grad()
def decode_tokens(nova_model, nova_cfg,
                  tokens: torch.Tensor, pts_norm: torch.Tensor,
                  device: torch.device, num_queries: int, seed: int) -> np.ndarray:
    torch.manual_seed(seed)
    encoder_data = {"tokens": tokens.to(device)}
    images  = torch.zeros(1, 1, 3, 1, 1, device=device)
    x_init  = torch.rand(1, num_queries, 3, device=device) * 2 - 1
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


def norm_to_world(pts_norm_np: np.ndarray, norm_factor: float,
                  c2w: np.ndarray) -> np.ndarray:
    pts_cam = pts_norm_np / 3.0 * norm_factor
    ones    = np.ones((len(pts_cam), 1), dtype=np.float32)
    return (c2w @ np.hstack([pts_cam, ones]).T).T[:, :3].astype(np.float32)


def main() -> None:
    args = parse_args()
    cfg  = OmegaConf.load(CFG_PATH)
    OmegaConf.set_struct(cfg, False)
    device = torch.device(args.device)

    val_starts_explicit = args.val_starts is not None
    val_root_auto_starts = args.val_root is not None and not val_starts_explicit
    val_starts = list(args.val_starts) if val_starts_explicit else [56]

    # ── resolve room configs: [(root, [train_starts]), ...] ───────────────────
    if args.rooms:
        room_configs = []
        for r in args.rooms:
            if ":" in r:
                root_str, starts_str = r.rsplit(":", 1)
                starts = [int(s) for s in starts_str.split(",")]
                root = Path(root_str)
            else:
                root = Path(r)
                starts = discover_cached_starts(root)
            room_configs.append((root, starts))
    else:
        if args.train_starts is None:
            single_train = discover_train_starts(args.data_root, val_starts, args.frame_stride)
        else:
            single_train = list(args.train_starts)
        room_configs = [(args.data_root, single_train)]

    n_rooms = len(room_configs)
    print(f"[setup] {n_rooms} room(s)  val={'auto' if val_root_auto_starts else val_starts}")
    for root, starts in room_configs:
        print(f"  root={root}  n_train={len(starts)}")

    # ── load all windows ───────────────────────────────────────────────────────
    train_metas, train_da3, train_z, train_room_labels = [], [], [], []
    for root, starts in room_configs:
        room_label = root.name
        for s in starts:
            meta, da3, z, _ = load_window(s, root)
            train_metas.append(meta)
            train_da3.append(da3)
            train_z.append(z[0])
            train_room_labels.append(room_label)
            print(f"  [{room_label}] train start={s:4d}  frames {meta['frame_ids'][0]}–{meta['frame_ids'][-1]}  "
                  f"norm_factor={float(meta['norm_factor']):.3f}m")

    val_metas, val_da3s, val_zs, val_pnorms, val_room_labels = [], [], [], [], []
    if args.val_rooms:
        val_configs = []
        for r in args.val_rooms:
            root_str, starts_str = r.rsplit(":", 1)
            val_configs.append((Path(root_str), [int(s) for s in starts_str.split(",")]))
    else:
        val_roots = (list(args.val_root) if args.val_root
                     else [root for root, _ in room_configs])
        val_configs = []
        for vroot in val_roots:
            starts = discover_cached_starts(vroot) if val_root_auto_starts else val_starts
            val_configs.append((vroot, starts))

    for vroot, starts in val_configs:
        room_label = vroot.name
        for vs in starts:
            vm, vd, vz, vp = load_window(vs, vroot)
            val_metas.append(vm); val_da3s.append(vd); val_zs.append(vz); val_pnorms.append(vp)
            val_room_labels.append(room_label)
            print(f"  [{room_label}] val   start={vs:4d}  "
                  f"frames {vm['frame_ids'][0]}–{vm['frame_ids'][-1]}  "
                  f"norm_factor={float(vm['norm_factor']):.3f}m")

    da3_train   = torch.cat(train_da3, dim=0)      # (N, T, D), kept on CPU
    zstar_train = torch.stack(train_z)             # (N, 768, 128), raw/no pre-alignment, kept on CPU
    N = da3_train.shape[0]

    # ── adapter ───────────────────────────────────────────────────────────────
    adapter = DA3ToNOVA3RAlignment(
        source_dim    = int(cfg.source_dim),
        hidden_dim    = int(cfg.hidden_dim),
        target_tokens = int(cfg.target_tokens),
        target_dim    = int(cfg.target_dim),
        depth         = int(cfg.depth),
        num_heads     = int(cfg.num_heads),
        drop          = 0.0,
    ).to(device)
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=args.lr)
    rng = torch.Generator().manual_seed(args.seed)
    batch_size = N if args.batch_size is None else min(int(args.batch_size), N)

    val_tag  = "vall" if val_root_auto_starts else "v" + "_".join(str(v) for v in val_starts)
    room_tag = f"{n_rooms}r"
    run_tag  = f"hungarian_online_{room_tag}_{val_tag}"
    if args.rrd_out is None:
        ckpt_path = (REPO_ROOT / "outputs" / "consecutive_windows" /
                     f"{run_tag}_best.pt")
    else:
        ckpt_path = args.rrd_out.with_name(f"{args.rrd_out.stem}_best.pt")
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    init_step = 0
    init_val_loss = float("inf")
    init_state = None
    saved_checkpoint = False

    def checkpoint_payload(state_dict, step: int, val_loss: float) -> dict:
        return {
            "state_dict": state_dict,
            "step": step,
            "val_loss": val_loss,
            "metadata": {
                "init_ckpt": str(args.init_ckpt) if args.init_ckpt is not None else None,
                "init_step": init_step,
                "init_val_loss": init_val_loss,
                "n_train_windows": N,
                "n_val_windows": len(val_metas),
                "train_configs": [
                    {"root": str(root), "starts": list(starts)}
                    for root, starts in room_configs
                ],
                "val_configs": [
                    {"root": str(root), "starts": list(starts)}
                    for root, starts in val_configs
                ],
                "frame_stride": args.frame_stride,
                "batch_size": batch_size,
                "lr": args.lr,
                "steps_requested": args.steps,
                "seed": args.seed,
            },
        }

    if args.init_ckpt is not None:
        init_data = torch.load(args.init_ckpt, map_location="cpu", weights_only=False)
        adapter.load_state_dict(init_data["state_dict"])
        init_step = int(init_data.get("step", 0))
        init_val_loss = float(init_data.get("val_loss", float("inf")))
        init_state = {k: v.cpu().clone() for k, v in adapter.state_dict().items()}
        print(f"[init] warm-start from {args.init_ckpt}  "
              f"(step={init_data.get('step','?')}  val_loss={init_data.get('val_loss','?')})")

    if args.load_ckpt is not None:
        # ── load checkpoint, skip training ────────────────────────────────────
        ckpt_data = torch.load(args.load_ckpt, map_location="cpu", weights_only=False)
        adapter.load_state_dict(ckpt_data["state_dict"])
        adapter.eval()
        best_step     = ckpt_data.get("step", -1)
        best_val_loss = ckpt_data.get("val_loss", float("nan"))
        best_state    = ckpt_data["state_dict"]
        print(f"[ckpt] loaded {args.load_ckpt}  step={best_step}  val_loss={best_val_loss:.6f}")
    else:
        # ── training loop ─────────────────────────────────────────────────────
        print(f"\n[train] {N} windows  {args.steps} steps  lr={args.lr}  "
              f"batch={batch_size}  (online Hungarian matching loss)")
        adapter.train()
        best_val_loss = init_val_loss
        best_step     = init_step
        best_state    = init_state

        for step in range(1, args.steps + 1):
            idx  = torch.randperm(N, generator=rng)[:batch_size]
            pred = adapter(da3_train[idx].to(device))
            loss = hungarian_mse(pred, zstar_train[idx].to(device))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            if step % args.log_every == 0 or step == 1 or step == args.steps:
                with torch.no_grad():
                    val_loss = float(np.mean([
                        hungarian_mse(adapter(vd.to(device)), vz.to(device)).item()
                        for vd, vz in zip(val_da3s, val_zs)
                    ]))
                marker = " ← best" if val_loss < best_val_loss else ""
                print(f"  step {step:5d}/{args.steps}  "
                      f"train_loss={loss.item():.6f}  val_loss={val_loss:.6f}{marker}", flush=True)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_step     = init_step + step
                    best_state    = {k: v.cpu().clone() for k, v in adapter.state_dict().items()}
                    torch.save(checkpoint_payload(best_state, best_step, best_val_loss), ckpt_path)
                    saved_checkpoint = True

        print(f"\n[best]  step {best_step}  val MSE {best_val_loss:.6f}")
        adapter.load_state_dict(best_state)
        adapter.eval()

        if saved_checkpoint:
            print(f"[best]  saved → {ckpt_path}")
        else:
            print(f"[best]  no new checkpoint saved; validation did not beat init val MSE {init_val_loss:.6f}")

    with torch.no_grad():
        val_pred_tokens = [adapter(vd.to(device)).cpu() for vd in val_da3s]

    with torch.no_grad():
        train_loss_chunks = []
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            pred_chunk = adapter(da3_train[start:end].to(device)).cpu()
            train_loss_chunks.append(
                hungarian_mse(pred_chunk, zstar_train[start:end]).item())
        final_train = float(np.mean(train_loss_chunks))
        final_val   = float(np.mean([
            hungarian_mse(vp, vz).item()
            for vp, vz in zip(val_pred_tokens, val_zs)
        ]))
    print(f"[result] train Hungarian MSE: {final_train:.6f}  val Hungarian MSE: {final_val:.6f}  "
          f"(best model from step {best_step})")

    # ── NOVA3R decode ──────────────────────────────────────────────────────────
    print("[decode] Loading NOVA3R …")
    ckpt = str(REPO_ROOT / "nova3r" / "checkpoints" / "scene_ae" / "checkpoint-last.pth")
    nova_model, nova_cfg = load_nova3r_model(ckpt, str(device))
    nova_model.eval()
    for p in nova_model.parameters():
        p.requires_grad_(False)
    OmegaConf.set_struct(nova_cfg, False)

    # ── Rerun ─────────────────────────────────────────────────────────────────
    import rerun as rr

    if args.rrd_out is None:
        rrd_out = REPO_ROOT / "outputs" / "consecutive_windows" / f"{run_tag}.rrd"
    else:
        rrd_out = args.rrd_out
    rrd_out.parent.mkdir(parents=True, exist_ok=True)

    rr.init("hungarian_windows", spawn=False)
    rr.save(str(rrd_out))
    rr.log("world", rr.ViewCoordinates.RDF, static=True)

    BLUE   = np.array([ 35, 115, 255], dtype=np.uint8)  # GT consensus
    RED    = np.array([230,  45,  55], dtype=np.uint8)  # adapter prediction
    GREEN  = np.array([ 45, 205,  90], dtype=np.uint8)  # input pts
    PURPLE = np.array([165,  85, 245], dtype=np.uint8)  # cameras

    def sub(pts: np.ndarray, n: int = 100_000) -> np.ndarray:
        if len(pts) <= n:
            return pts
        return pts[np.random.default_rng(0).choice(len(pts), n, replace=False)]

    axis_index = {"x": 0, "y": 1, "z": 2}[args.shade_axis]

    def shaded_colors(pts: np.ndarray, base: np.ndarray,
                      axis_min: float, axis_max: float) -> np.ndarray:
        """Keep a stable class color, varying brightness along one world axis."""
        values = pts[:, axis_index]
        t = np.clip((values - axis_min) / (axis_max - axis_min + 1e-8), 0, 1)
        shade = 0.42 + 0.58 * t
        return np.clip(base[None, :].astype(np.float32) * shade[:, None], 0, 255).astype(np.uint8)

    def rotate_for_viz(pts: np.ndarray) -> np.ndarray:
        if args.rot90_axis == "none":
            return pts
        rotations = {
            "x":  np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float32),
            "-x": np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=np.float32),
            "y":  np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]], dtype=np.float32),
            "-y": np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]], dtype=np.float32),
            "z":  np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float32),
            "-z": np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]], dtype=np.float32),
        }
        return pts @ rotations[args.rot90_axis].T

    n_val_total = len(val_metas)
    for i, (vm, vz_cpu, vp, vpt, vroom) in enumerate(
            zip(val_metas, val_zs, val_pnorms, val_pred_tokens, val_room_labels)):
        vs = int(vm.get("start_idx", val_starts[i % len(val_starts)]))
        rr.set_time("val_window", sequence=i)

        nf  = float(vm["norm_factor"])
        c2w = vm["poses_c2w"][0].numpy()

        print(f"[decode] [{vroom}] val start={vs}  GT …")
        gt_norm   = decode_tokens(nova_model, nova_cfg, vz_cpu.cpu(), vp,
                                  device, args.num_queries, seed=args.seed)
        print(f"[decode] [{vroom}] val start={vs}  Pred …")
        pred_norm = decode_tokens(nova_model, nova_cfg, vpt, vp,
                                  device, args.num_queries, seed=args.seed)

        gt_world   = norm_to_world(gt_norm,   nf, c2w)
        pred_world = norm_to_world(pred_norm, nf, c2w)
        input_cam  = vp[0].numpy() / 3.0 * nf
        ones       = np.ones((len(input_cam), 1), dtype=np.float32)
        input_world = (c2w @ np.hstack([input_cam, ones]).T).T[:, :3]
        camera_world = vm["poses_c2w"][:, :3, 3].numpy()

        print(f"[decode] GT centroid:   {gt_world.mean(0).round(3)}")
        print(f"[decode] Pred centroid: {pred_world.mean(0).round(3)}")

        gt_world = rotate_for_viz(gt_world)
        pred_world = rotate_for_viz(pred_world)
        input_world = rotate_for_viz(input_world)
        camera_world = rotate_for_viz(camera_world)

        shade_values = np.concatenate([
            gt_world[:, axis_index],
            pred_world[:, axis_index],
            input_world[:, axis_index],
            camera_world[:, axis_index],
        ])
        shade_min, shade_max = shade_values.min(), shade_values.max()

        gt_sub   = sub(gt_world)
        pred_sub = sub(pred_world)
        inp_sub  = sub(input_world, n=30_000)

        rr.log("world/gt_consensus",
               rr.Points3D(gt_sub,
                           colors=shaded_colors(gt_sub, BLUE, shade_min, shade_max),
                           radii=0.012))
        rr.log("world/adapter_pred",
               rr.Points3D(pred_sub,
                           colors=shaded_colors(pred_sub, RED, shade_min, shade_max),
                           radii=0.012))
        rr.log("world/input_pts",
               rr.Points3D(inp_sub,
                           colors=shaded_colors(inp_sub, GREEN, shade_min, shade_max),
                           radii=0.008))
        rr.log("world/cameras",
               rr.Points3D(camera_world,
                           colors=shaded_colors(camera_world, PURPLE, shade_min, shade_max),
                           radii=0.04))

        val_fr = f"{vm['frame_ids'][0]}-{vm['frame_ids'][-1]}"
        rr.log("legend", rr.TextDocument(
            f"# [{vroom}] val start={vs}  ({i+1}/{n_val_total})\n\n"
            "- **Blue** GT consensus\n"
            "- **Red** adapter prediction\n"
            "- **Green** input points\n"
            "- **Purple** cameras\n"
            f"- Shade varies by world `{args.shade_axis}` axis\n\n"
            f"**Visualization rotation:** `{args.rot90_axis}`\n\n"
            f"**Val frames:** {val_fr}\n\n"
            f"N={N} train windows ({n_rooms} rooms)  |  Steps={args.steps}  |  LR={args.lr}\n\n"
            f"Train MSE: {final_train:.4f}  |  Val MSE (mean): {final_val:.4f}",
            media_type=rr.MediaType.MARKDOWN,
        ))

    print(f"\n[rerun] saved → {rrd_out}")
    print(f"  open with:  rerun {rrd_out}")


if __name__ == "__main__":
    main()
