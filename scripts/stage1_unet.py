#!/usr/bin/env python3
"""Stage 1 (tsdf.md): deterministic 3D-UNet TSDF completion.

Input : 5 channels on a 128^3 grid  [partial TSDF, mask one-hot(observed/free/unknown), confidence]
Output: 1 channel  = completed TSDF (tanh * band)
Loss  : L1 vs GT TSDF, MASKED to `unknown` voxels only, voxels near the surface weighted 5x.

`--overfit` runs the spec's verification gate: fit ONE crop and check the loss collapses and the
predicted TSDF reproduces the GT on unknown voxels. This validates the data tensors + net + loss
+ masking before any real training (which later swaps the single crop for a crop dataset, and
SCRREAM meshes for Replica).
"""
import argparse, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OBSERVED, FREE, UNKNOWN = 0, 1, 2


class ResBlock(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.n1 = nn.GroupNorm(8, cin); self.c1 = nn.Conv3d(cin, cout, 3, padding=1)
        self.n2 = nn.GroupNorm(8, cout); self.c2 = nn.Conv3d(cout, cout, 3, padding=1)
        self.skip = nn.Conv3d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x):
        h = self.c1(F.silu(self.n1(x)))
        h = self.c2(F.silu(self.n2(h)))
        return h + self.skip(x)


class OcclusionUNet(nn.Module):
    """Dense 3D UNet, 16->32->64->128->256 over 128^3->8^3, two res blocks per level."""
    def __init__(self, in_ch=5, base=16, band=0.10):
        super().__init__()
        self.band = band
        chs = [base, base * 2, base * 4, base * 8, base * 16]      # 16,32,64,128,256
        self.stem = nn.Conv3d(in_ch, chs[0], 3, padding=1)
        self.enc = nn.ModuleList(); self.down = nn.ModuleList()
        for i in range(4):
            self.enc.append(nn.Sequential(ResBlock(chs[i], chs[i]), ResBlock(chs[i], chs[i])))
            self.down.append(nn.Conv3d(chs[i], chs[i + 1], 3, stride=2, padding=1))
        self.mid = nn.Sequential(ResBlock(chs[4], chs[4]), ResBlock(chs[4], chs[4]))
        self.up = nn.ModuleList(); self.dec = nn.ModuleList()
        for i in range(4, 0, -1):
            self.up.append(nn.ConvTranspose3d(chs[i], chs[i - 1], 2, stride=2))
            self.dec.append(nn.Sequential(ResBlock(chs[i - 1] * 2, chs[i - 1]), ResBlock(chs[i - 1], chs[i - 1])))
        self.head = nn.Conv3d(chs[0], 1, 1)

    def forward(self, x):
        x = self.stem(x); skips = []
        for enc, down in zip(self.enc, self.down):
            x = enc(x); skips.append(x); x = down(x)
        x = self.mid(x)
        for up, dec, s in zip(self.up, self.dec, reversed(skips)):
            x = up(x); x = dec(torch.cat([x, s], 1))
        return torch.tanh(self.head(x)) * self.band


def build_input(gt_tsdf, partial_tsdf, mask, band):
    """5-channel network input from the pipeline tensors."""
    pt = torch.from_numpy(partial_tsdf.astype(np.float32)) / band          # [-1,1]
    m = torch.from_numpy(mask.astype(np.int64))
    onehot = F.one_hot(m, 3).permute(3, 0, 1, 2).float()                   # 3 x D^3
    conf = torch.ones(1, *pt.shape)                                        # constant at train time
    return torch.cat([pt[None], onehot, conf], 0)                          # 5 x D^3


def masked_surface_l1(pred, gt, mask, surf_band, surf_w=5.0):
    """L1 on `unknown` voxels only; voxels genuinely NEAR the surface (|gt| < surf_band) weighted
    surf_w higher. surf_band is strictly inside the truncation so fp16-saturated far voxels
    (stored at ~0.0999 for a 0.10 band) are NOT counted as 'near surface'."""
    m = (mask == UNKNOWN).float()
    w = torch.where(gt.abs() < surf_band, torch.full_like(gt, surf_w), torch.ones_like(gt)) * m
    return (w * (pred - gt).abs()).sum() / w.sum().clamp_min(1.0)


def load_grid(npz):
    d = np.load(npz)
    return (d["gt_tsdf"].astype(np.float32), d["partial_tsdf"].astype(np.float32),
            d["mask"].astype(np.int8), float(d["voxel"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", default="/usr/prakt/s0016/vc3r/outputs/tsdf_pipeline/office4_win0/grid.npz")
    ap.add_argument("--overfit", action="store_true")
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--band", type=float, default=0.10)
    ap.add_argument("--base", type=int, default=16)
    ap.add_argument("--ckpt-out", default="/usr/prakt/s0016/vc3r/outputs/consecutive_windows/stage1_overfit.pt")
    args = ap.parse_args()

    gt_np, pt_np, mask_np, voxel = load_grid(args.grid)
    band = args.band
    surf_band = band - voxel                       # 'near surface' shell, margin excludes fp16 saturation
    inp = build_input(gt_np, pt_np, mask_np, band).unsqueeze(0).to(DEV)
    gt = torch.from_numpy(gt_np).to(DEV)[None, None]
    mask = torch.from_numpy(mask_np.astype(np.int64)).to(DEV)[None, None]
    n_unknown = int((mask == UNKNOWN).sum())
    n_unk_surf = int(((mask == UNKNOWN) & (gt.abs() < surf_band)).sum())
    print(f"[data] grid {gt_np.shape} voxel={voxel*100:.0f}cm  unknown={n_unknown:,}  "
          f"unknown-near-surface(target)={n_unk_surf:,}", flush=True)

    net = OcclusionUNet(in_ch=5, base=args.base, band=band).to(DEV)
    print(f"[model] OcclusionUNet params={sum(p.numel() for p in net.parameters())/1e6:.2f}M", flush=True)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=0.0)

    # baseline: predicting the partial TSDF everywhere (no completion) — the number to beat
    with torch.no_grad():
        pt_t = torch.from_numpy(pt_np).to(DEV)[None, None]
        base_l1 = masked_surface_l1(pt_t, gt, mask, surf_band).item()
    print(f"[baseline] surface-L1 of PARTIAL tsdf on unknown = {base_l1*100:.3f} cm-equiv", flush=True)

    net.train(); t0 = time.time()
    for step in range(1, args.steps + 1):
        pred = net(inp)
        loss = masked_surface_l1(pred, gt, mask, surf_band)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        if step % 50 == 0 or step == 1:
            with torch.no_grad():
                m = (mask == UNKNOWN) & (gt.abs() < surf_band)
                surf_mae = (pred - gt).abs()[m].mean().item()
                sign_acc = ((pred.sign() == gt.sign())[m].float().mean().item())
            el = time.time() - t0
            print(f"  step {step:4d}/{args.steps} loss={loss.item()*100:.3f}  "
                  f"unk-surf MAE={surf_mae*100:.3f}cm  sign-acc={sign_acc*100:.1f}%  "
                  f"t={el:.0f}s", flush=True)

    Path(args.ckpt_out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": net.state_dict(), "args": vars(args)}, args.ckpt_out)
    print(f"[save] {args.ckpt_out}", flush=True)
    print("[verdict] overfit PASSES if unk-surf MAE << baseline and sign-acc -> ~100%.", flush=True)


if __name__ == "__main__":
    main()
