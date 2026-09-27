#!/usr/bin/env python3
"""DIAGNOSTIC #2 (generalization): does decoder fine-tuning transfer to a HELD-OUT scene?

The overfit probe (diag_decoder_overfit.py) proved z* can support ~22% crisper geometry when the
decoder is fit to a scene's OWN (z*, GT). This asks the question that matters for the real pipeline:
train the decoder (LoRA on pts3d_head) on office0-3 + room0-2 and measure the oracle on HELD-OUT
office4. Encoder (aggregator) frozen so z* is fixed (adapter target unchanged). FM-to-GT loss.

LoRA (rank-r on every plain nn.Linear in pts3d_head; MHA out_proj/in_proj skipped) preserves the
pretrained generative prior and limits overfitting on the small scene set. On finish, LoRA deltas are
merged into the base weights and a CLEAN pts3d_head state_dict is saved for a real stitch --decoder-ckpt
eval on office4.
"""
import argparse, math, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from scipy.spatial import cKDTree

REPO = Path("/usr/prakt/s0016/vc3r")
NOVA = REPO / "nova3r_lib"
for _p in [str(NOVA / "third_party"), str(NOVA), str(NOVA / "demo")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)
from demo_nova3r import load_model as load_nova3r_model            # noqa: E402
from nova3r.models.model_wrapper import BatchModelWrapper          # noqa: E402
from nova3r.flow_matching.solver import ODESolver                  # noqa: E402

DEV = torch.device("cuda")
NOVA_CKPT = REPO / "checkpoints" / "nova3r" / "scene_ae" / "checkpoint-last.pth"
CROOT = REPO / "scripts/data/fc_nf16_span24_100_l13_complete"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train-rooms", nargs="+",
                   default=["office0", "office1", "office2", "office3", "room0", "room1", "room2"])
    p.add_argument("--val-room", default="office4")
    p.add_argument("--max-per-room", type=int, default=60)
    p.add_argument("--eval-windows", type=int, default=16)
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--batch-windows", type=int, default=6)
    p.add_argument("--query-points", type=int, default=4096)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--lora-rank", type=int, default=16)
    p.add_argument("--lora-alpha", type=float, default=32.0)
    p.add_argument("--eval-queries", type=int, default=50000)
    p.add_argument("--log-every", type=int, default=250)
    p.add_argument("--head-out", default=str(REPO / "outputs/consecutive_windows/diag_decoder_lora_head.pt"))
    return p.parse_args()


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int, alpha: float):
        super().__init__()
        self.base = base
        self.base.weight.requires_grad_(False)
        if self.base.bias is not None:
            self.base.bias.requires_grad_(False)
        self.r, self.scale = r, alpha / r
        self.A = nn.Parameter(torch.zeros(r, base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, r))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))   # B stays 0 -> delta starts at 0

    def forward(self, x):
        return self.base(x) + (x @ self.A.t() @ self.B.t()) * self.scale

    @torch.no_grad()
    def merge(self):
        self.base.weight.add_((self.B @ self.A) * self.scale)


def inject_lora(module, r, alpha):
    """Wrap every plain nn.Linear (skip MHA's NonDynamicallyQuantizableLinear out_proj)."""
    n = 0
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear) and not isinstance(
                child, nn.modules.linear.NonDynamicallyQuantizableLinear):
            setattr(module, name, LoRALinear(child, r, alpha)); n += 1
        else:
            n += inject_lora(child, r, alpha)
    return n


def fm_to_gt_velocity(surf_pts, n_q):
    B, K = surf_pts.shape[0], surf_pts.shape[1]
    sel = torch.randint(K, (B, n_q), device=surf_pts.device)
    x1 = torch.gather(surf_pts, 1, sel.unsqueeze(-1).expand(B, n_q, 3))
    x0 = torch.rand(B, n_q, 3, device=surf_pts.device) * 2 - 1
    t = torch.rand(B, 1, 1, device=surf_pts.device)
    a, s = torch.sin(math.pi / 2 * t), torch.cos(math.pi / 2 * t)
    da, ds = (math.pi / 2) * torch.cos(math.pi / 2 * t), -(math.pi / 2) * torch.sin(math.pi / 2 * t)
    return s * x0 + a * x1, t.view(B, 1).expand(B, n_q), ds * x0 + da * x1


@torch.no_grad()
def oracle_decode(nova, ncfg, z_star, pts_norm, num_q, seed=0):
    torch.manual_seed(seed)
    x_init = torch.rand(1, num_q, 3, device=DEV) * 2 - 1
    solver = ODESolver(velocity_model=BatchModelWrapper(model=nova))
    step = ncfg.get("fm_step_size", 0.04)
    T = torch.linspace(0, 1, int(1 // step)).to(DEV)
    with torch.amp.autocast("cuda", enabled=False):
        sol = solver.sample(time_grid=T, x_init=x_init, method=ncfg.get("fm_sampling", "euler"),
                            step_size=step, return_intermediates=False,
                            images=torch.zeros(1, 1, 3, 1, 1, device=DEV),
                            token_mask=None, encoder_data={"tokens": z_star.to(DEV)},
                            pointmaps=pts_norm.to(DEV))
    return (sol[-1] if isinstance(sol, list) else sol)[0].cpu().float().numpy()


def chamfer_norm(pred, gt):
    tp, tg = cKDTree(pred), cKDTree(gt)
    d_pg, _ = tg.query(pred, k=1)
    d_gp, _ = tp.query(gt, k=1)
    return float((d_pg.mean() + d_gp.mean()) / 2)


def eval_oracle(nova, ncfg, windows, num_q, tag):
    accs = [chamfer_norm(oracle_decode(nova, ncfg, z, pn, num_q), pn[0].numpy()) for z, pn in windows]
    print(f"    [{tag}] q={num_q} HELD-OUT oracle Chamfer(norm) mean={np.mean(accs):.4f} "
          f"median={np.median(accs):.4f}  (n={len(accs)})", flush=True)
    return float(np.mean(accs))


def load_room(room, cap):
    d = CROOT / room
    wds = sorted([w for w in d.iterdir() if w.is_dir()])[:cap]
    Z = [torch.load(w / "z_star_online_mean.pt", map_location="cpu", weights_only=True).float() for w in wds]
    PN = [torch.load(w / "pts_norm.pt", map_location="cpu", weights_only=True).float() for w in wds]
    return Z, PN


def main():
    args = parse_args()
    Z, PN = [], []
    for r in args.train_rooms:
        z, pn = load_room(r, args.max_per_room)
        Z += z; PN += pn
        print(f"[data] train {r}: {len(z)} windows", flush=True)
    N = len(Z)
    vz, vpn = load_room(args.val_room, args.eval_windows)
    eval_set = list(zip(vz, vpn))
    print(f"[data] {N} train windows | held-out {args.val_room}: {len(eval_set)} eval windows", flush=True)

    print("[load] NOVA3R scene_ae", flush=True)
    nova, ncfg = load_nova3r_model(str(NOVA_CKPT), str(DEV))
    nova.eval()
    for p in nova.parameters():
        p.requires_grad_(False)
    n_wrapped = inject_lora(nova.pts3d_head, args.lora_rank, args.lora_alpha)
    nova.pts3d_head.to(DEV)
    lora_params = [p for p in nova.pts3d_head.parameters() if p.requires_grad]
    print(f"[lora] wrapped {n_wrapped} Linear layers, rank={args.lora_rank} "
          f"-> {sum(p.numel() for p in lora_params)/1e6:.2f}M trainable params", flush=True)

    print("[baseline] held-out oracle before LoRA FT:")
    eval_oracle(nova, ncfg, eval_set, args.eval_queries, "before")

    opt = torch.optim.AdamW(lora_params, lr=args.lr, weight_decay=0.0)
    rng = torch.Generator().manual_seed(0)
    t0 = time.time()
    for step in range(1, args.steps + 1):
        idx = torch.randperm(N, generator=rng)[:args.batch_windows].tolist()
        z = torch.cat([Z[i] for i in idx], 0).to(DEV)
        K = min(PN[i].shape[1] for i in idx)
        surf = torch.cat([PN[i][:, :K] for i in idx], 0).to(DEV)
        xt, tq, v_tgt = fm_to_gt_velocity(surf, args.query_points)
        img0 = torch.zeros(z.shape[0], 1, 3, 1, 1, device=DEV)
        with torch.amp.autocast("cuda", enabled=False):
            v_pred = nova._decode(tokens=z, images=img0, query_points=xt, timestep=tq)["pts3d_xyz"]
            loss = (v_pred - v_tgt).square().mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(lora_params, 1.0)
        opt.step()
        if step % args.log_every == 0 or step == 1:
            el = time.time() - t0
            print(f"  step {step:5d}/{args.steps} fm_loss={loss.item():.5f} "
                  f"t={el:.0f}s ({el/step:.3f}s/it)", flush=True)
            if step % (args.log_every * 4) == 0:
                eval_oracle(nova, ncfg, eval_set, args.eval_queries, f"step{step}")

    print("[after] held-out oracle after LoRA FT:")
    eval_oracle(nova, ncfg, eval_set, args.eval_queries, "after")

    for m in nova.pts3d_head.modules():                 # fold LoRA into base weights
        if isinstance(m, LoRALinear):
            m.merge()
    sd = nova.pts3d_head.state_dict()
    clean = {k.replace(".base.", "."): v.cpu() for k, v in sd.items()
             if not (k.endswith(".A") or k.endswith(".B"))}
    out = Path(args.head_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(clean, out)
    print(f"[out] merged LoRA head ({len(clean)} tensors) -> {out}", flush=True)


if __name__ == "__main__":
    main()
