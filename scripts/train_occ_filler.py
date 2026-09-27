#!/usr/bin/env python3
"""OCCLUSION FILLER PoC: conditional point flow-matching that GENERATES the complete surface
from noise, conditioned on a PARTIAL (occluded) point cloud. Geometry-supervised end to end
(loss on points), so the token!=geometry gap that sank every latent-space attempt cannot occur.

First experiment: synthesize occlusion by masking the cached COMPLETE pts_norm (drop a random
half-space, mimicking a viewpoint). Train partial->complete on office0-3+room0-2, hold out
office4. Eval: given a held-out window's partial, sample the completion and measure whether it
fills the removed region.

    x1 = complete surface (pts_norm),  x0 = moment-matched Gaussian noise,
    x_t = (1-t) x0 + t x1,  v* = x1 - x0,   v_theta(x_t, t | partial).
If it fills synthetic holes, swap the synthetic partial for the real DA3-encode partial next.
"""
import argparse, math, sys, time
from pathlib import Path
import numpy as np, torch
from scipy.spatial import cKDTree

REPO = Path("/usr/prakt/s0016/vc3r")
sys.path.insert(0, str(REPO / "da3" / "src"))
from vc3r.alignment import CrossAttentionBlock  # noqa: E402
DEV = torch.device("cuda")
CROOT = REPO / "scripts/data/fc_nf16_span24_100_l13_complete"


def fourier(x, bands):
    f = (2.0 ** torch.arange(bands, device=x.device, dtype=x.dtype)) * math.pi
    a = x[..., None] * f
    return torch.cat([a.sin(), a.cos()], -1).reshape(*x.shape[:-1], -1)


class OcclusionFiller(torch.nn.Module):
    def __init__(self, hidden=384, depth=6, heads=8, fb=10, tb=64):
        super().__init__()
        self.fb, self.tb = fb, tb
        self.point_in = torch.nn.Linear(3 * 2 * fb, hidden)
        self.cond_in = torch.nn.Linear(3 * 2 * fb, hidden)
        self.time_in = torch.nn.Linear(2 * tb, hidden)
        self.blocks = torch.nn.ModuleList([CrossAttentionBlock(hidden, heads) for _ in range(depth)])
        self.norm = torch.nn.LayerNorm(hidden)
        self.out = torch.nn.Linear(hidden, 3)
        self.out.weight.data.mul_(0.1); self.out.bias.data.zero_()

    def forward(self, x_t, t, cond):
        mem = self.cond_in(fourier(cond, self.fb))
        h = self.point_in(fourier(x_t, self.fb)) + self.time_in(fourier(t[..., None], self.tb)).unsqueeze(1)
        for blk in self.blocks:
            h = blk(h, mem)
        return self.out(self.norm(h))


def occlude(pts, keep_lo=0.5, keep_hi=0.75, gen=None):
    """Drop a random half-space of `pts` -> partial view. pts [K,3] -> [K',3] mask (bool)."""
    u = torch.randn(3, generator=gen); u = u / u.norm()
    proj = pts @ u
    keep = keep_lo + (keep_hi - keep_lo) * torch.rand(1, generator=gen).item()
    thr = torch.quantile(proj, keep)
    return proj < thr                                    # keep the near side


def take(pts, n, gen):
    if pts.shape[0] >= n:
        idx = torch.randperm(pts.shape[0], generator=gen)[:n]
    else:
        idx = torch.cat([torch.arange(pts.shape[0]), torch.randint(pts.shape[0], (n - pts.shape[0],), generator=gen)])
    return pts[idx]


def moments(x):  # per-cloud mean/std for the noise prior
    return x.mean(0, keepdim=True), x.std(0, keepdim=True).clamp_min(1e-3)


@torch.no_grad()
def sample(model, cond, n, steps=24):
    mu, sd = moments(cond[0])
    x = mu + sd * torch.randn(1, n, 3, device=DEV)
    ts = torch.linspace(0, 1, steps + 1, device=DEV)
    for i in range(steps):
        dt = ts[i + 1] - ts[i]
        k1 = model(x, ts[i].expand(1), cond)
        xm = x + 0.5 * dt * k1
        x = x + dt * model(xm, (ts[i] + 0.5 * dt).expand(1), cond)
    return x[0]


def prf(pred, gt, thr):
    d_pg = cKDTree(gt).query(pred)[0]; d_gp = cKDTree(pred).query(gt)[0]
    p = float((d_pg < thr).mean()); r = float((d_gp < thr).mean())
    return 2 * p * r / (p + r + 1e-9), p, r


def load_pn(rooms, cap):
    out = []
    for room in rooms:
        for d in sorted([x for x in (CROOT / room).iterdir() if x.is_dir()])[:cap]:
            out.append(torch.load(d / "pts_norm.pt", weights_only=True)[0].float())
    return out


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--rooms", nargs="+", default=["office0", "office1", "office2", "office3", "room0", "room1", "room2"])
    p.add_argument("--val-room", default="office4")
    p.add_argument("--cap", type=int, default=None)
    p.add_argument("--val-windows", type=int, default=8)
    p.add_argument("--steps", type=int, default=8000)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--n-pts", type=int, default=2048)
    p.add_argument("--cond-pts", type=int, default=1024)
    p.add_argument("--hidden", type=int, default=384)
    p.add_argument("--depth", type=int, default=6)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--sample-steps", type=int, default=24)
    p.add_argument("--eval-every", type=int, default=2000)
    p.add_argument("--thr", type=float, default=0.03)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--ckpt-out", type=Path, default=REPO / "outputs/consecutive_windows/occ_filler_poc.pt")
    return p.parse_args()


def main():
    a = parse(); torch.manual_seed(a.seed)
    train = load_pn(a.rooms, a.cap)
    val = load_pn([a.val_room], None)[:a.val_windows]
    print(f"[data] train={len(train)} windows  val={len(val)}", flush=True)
    model = OcclusionFiller(hidden=a.hidden, depth=a.depth).to(DEV)
    print(f"[model] OcclusionFiller hidden={a.hidden} depth={a.depth} "
          f"params={sum(p.numel() for p in model.parameters())/1e6:.2f}M", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.steps, eta_min=1e-6)
    gen = torch.Generator().manual_seed(a.seed)

    @torch.no_grad()
    def evaluate(step):
        model.eval()
        agg = {"PARTIAL": [], "FILLED": []}
        for pn in val:
            m = occlude(pn, gen=torch.Generator().manual_seed(123))
            partial = take(pn[m], a.cond_pts, torch.Generator().manual_seed(1)).unsqueeze(0).to(DEV)
            comp = pn.numpy()
            filled = sample(model, partial, a.n_pts, a.sample_steps).cpu().numpy()
            agg["PARTIAL"].append(prf(partial[0].cpu().numpy(), comp, a.thr))
            agg["FILLED"].append(prf(filled, comp, a.thr))
        print(f"\n[eval@{step}] held-out {a.val_room} ({len(val)} win, thr={a.thr} normalized, vs COMPLETE):")
        for k in ["PARTIAL", "FILLED"]:
            A = np.array(agg[k]); print(f"    {k:8s} F={A[:,0].mean():.3f}  prec={A[:,1].mean():.3f}  rec={A[:,2].mean():.3f}", flush=True)
        model.train()

    N = len(train); t0 = time.time(); model.train()
    for step in range(1, a.steps + 1):
        idx = torch.randint(0, N, (a.batch_size,), generator=gen).tolist()
        x1 = torch.empty(a.batch_size, a.n_pts, 3); cond = torch.empty(a.batch_size, a.cond_pts, 3)
        for b, i in enumerate(idx):
            pn = train[i]
            x1[b] = take(pn, a.n_pts, gen)
            cond[b] = take(pn[occlude(pn, gen=gen)], a.cond_pts, gen)
        x1 = x1.to(DEV); cond = cond.to(DEV)
        mu = x1.mean(1, keepdim=True); sd = x1.std(1, keepdim=True).clamp_min(1e-3)
        x0 = mu + sd * torch.randn_like(x1)
        t = torch.rand(a.batch_size, device=DEV)
        xt = (1 - t)[:, None, None] * x0 + t[:, None, None] * x1
        loss = ((model(xt, t, cond) - (x1 - x0)) ** 2).mean()
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sch.step()
        if step % 200 == 0 or step == 1:
            el = time.time() - t0
            print(f"[train] step {step}/{a.steps} loss={loss.item():.4f} lr={sch.get_last_lr()[0]:.2e} "
                  f"t={el:.0f}s ({el/step:.2f}s/it)", flush=True)
        if step % a.eval_every == 0:
            evaluate(step)
    evaluate(a.steps)
    a.ckpt_out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "args": vars(a)}, a.ckpt_out)
    print(f"[save] {a.ckpt_out}", flush=True)


if __name__ == "__main__":
    main()
