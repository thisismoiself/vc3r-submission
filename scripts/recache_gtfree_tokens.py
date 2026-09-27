#!/usr/bin/env python3
"""Regenerate ONLY da3_tokens.pt for every window with the GT-FREE camera path
(cam_token=None, i.e. no GT poses fed to DA3 — the same path DA3 uses when
extrinsics are not supplied). z*/pts_norm/meta (the GT complete-target) are reused
by symlink, so the target is unchanged; only the adapter INPUT becomes GT-free.

Reads windows from the existing complete cache (meta.pt gives frame_ids + flip),
writes a parallel cache dir. Resumable: skips windows whose da3_tokens.pt exists.
"""
import argparse, json, sys, types
from pathlib import Path
import numpy as np, torch
from PIL import Image

REPO = Path("/usr/prakt/s0016/vc3r")
sys.path.insert(0, str(REPO / "experiments" / "overfit_8frames"))
sys.path.insert(0, str(REPO / "da3" / "src"))
for _m in ("pycolmap",): sys.modules.setdefault(_m, types.ModuleType(_m))
from multi_scene_train import load_da3_model, extract_da3_tokens  # noqa: E402

X_FLIP = torch.diag(torch.tensor([-1., 1., 1., 1.]))


def build_k_proc(replica_root, img_w, img_h):
    cam = json.load(open(Path(replica_root) / "cam_params.json"))["camera"]
    h_nat, w_nat = int(cam["h"]), int(cam["w"])
    k = torch.tensor([[cam["fx"], 0, cam["cx"]], [0, cam["fy"], cam["cy"]], [0, 0, 1.0]])
    k[0] *= img_w / w_nat; k[1] *= img_h / h_nat
    return k

REUSE = ["z_star_online_mean.pt", "z_star_online_var.pt", "z_star_online_var_eff.pt",
         "z_star_online_samples.pt", "pts_norm.pt", "meta.pt"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-cache", default="scripts/data/fc_nf16_span24_100_l13_complete")
    ap.add_argument("--dst-cache", default="scripts/data/fc_nf16_span24_100_l13_complete_gtfree")
    ap.add_argument("--replica-root", default="/usr/prakt/s0016/Replica")
    ap.add_argument("--rooms", nargs="+",
                    default=["office0", "office1", "office2", "office3", "room0", "room1", "room2", "office4"])
    ap.add_argument("--da3-model", default="depth-anything/DA3-LARGE-1.1")
    ap.add_argument("--da3-layer", type=int, nargs="+", default=[1, 3])
    ap.add_argument("--da3-max-tokens", type=int, default=2048)
    ap.add_argument("--image-height", type=int, default=392)
    ap.add_argument("--image-width", type=int, default=518)
    ap.add_argument("--max-per-room", type=int, default=None, help="cap windows/room (smoke test)")
    ap.add_argument("--pose-source", choices=["none", "da3pred"], default="none",
                    help="none=cam_token=None (pose-free); da3pred=DA3-PREDICTED poses fed to cam_enc.")
    args = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    da3 = load_da3_model(args.da3_model, dev)
    src_root, dst_root = REPO / args.src_cache, REPO / args.dst_cache
    dummy_K = torch.eye(3).unsqueeze(0)  # unused when use_poses=False

    api, k_proc = None, None
    if args.pose_source == "da3pred":
        from depth_anything_3.api import DepthAnything3
        api = DepthAnything3.from_pretrained(args.da3_model).to(dev).eval()
        k_proc = build_k_proc(args.replica_root, args.image_width, args.image_height)
        print(f"[da3pred] k_proc=\n{k_proc.numpy()}", flush=True)

    def load_rgb(room, fid):
        p = Path(args.replica_root) / room / "results" / f"frame{fid:06d}.jpg"
        img = Image.open(p).convert("RGB").resize((args.image_width, args.image_height), Image.BILINEAR)
        return torch.from_numpy(np.array(img, np.float32)).permute(2, 0, 1) / 255.0

    total, done, skipped = 0, 0, 0
    for room in args.rooms:
        src_room = src_root / room
        if not src_room.is_dir():
            print(f"[skip] {room}: no src dir", flush=True); continue
        wins = sorted([d for d in src_room.iterdir() if d.is_dir()])
        if args.max_per_room:
            wins = wins[:args.max_per_room]
        print(f"[{room}] {len(wins)} windows", flush=True)
        for wd in wins:
            total += 1
            dst = dst_root / room / wd.name
            dst.mkdir(parents=True, exist_ok=True)
            tok_path = dst / "da3_tokens.pt"
            # reuse target files by symlink (idempotent)
            for f in REUSE:
                lp = dst / f
                if not lp.exists() and (wd / f).exists():
                    lp.symlink_to((wd / f).resolve())
            if tok_path.exists():
                skipped += 1; continue
            meta = torch.load(wd / "meta.pt", weights_only=False)
            fids, flip = meta["frame_ids"], bool(meta.get("flip", False))
            images = torch.stack([load_rgb(room, int(f)) for f in fids])
            if flip:
                images = torch.flip(images, dims=[-1])
            n = len(fids)
            with torch.no_grad():
                if args.pose_source == "da3pred":
                    # predict poses on the ORIGINAL (unflipped) frames, then flip via X_FLIP
                    paths = [str(Path(args.replica_root) / room / "results" / f"frame{int(f):06d}.jpg") for f in fids]
                    p = api.inference(image=paths, extrinsics=None, intrinsics=None, process_res=args.image_height)
                    ex = np.asarray(p.extrinsics)                       # (N,3,4) w2c
                    if ex.shape[-2:] == (3, 4):
                        bot = np.tile(np.array([0, 0, 0, 1], np.float32), (n, 1, 1))
                        ex = np.concatenate([ex, bot], axis=1)
                    c2w = torch.from_numpy(np.linalg.inv(ex)).float()  # (N,4,4)
                    if flip:
                        c2w = c2w @ X_FLIP
                    tok = extract_da3_tokens(da3, images, c2w, k_proc.repeat(n, 1, 1),
                                             layer_idx=args.da3_layer, max_tokens=args.da3_max_tokens,
                                             device=dev, use_poses=True)
                else:
                    tok = extract_da3_tokens(da3, images, torch.eye(4).repeat(n, 1, 1),
                                             dummy_K.repeat(n, 1, 1),
                                             layer_idx=args.da3_layer, max_tokens=args.da3_max_tokens,
                                             device=dev, use_poses=False)
            torch.save(tok, tok_path)
            done += 1
            if done % 25 == 0:
                print(f"  [{room}] {done} done (skip {skipped}) last={wd.name} tok={tuple(tok.shape)}", flush=True)
    print(f"DONE: total={total} written={done} skipped={skipped} -> {dst_root}", flush=True)


if __name__ == "__main__":
    main()
