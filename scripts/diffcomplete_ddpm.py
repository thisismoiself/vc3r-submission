#!/usr/bin/env python3
"""DiffComplete-style DDPM for TSDF completion (diffusion_training_spec.md, Part 2-3 & 6).

The model in one paragraph, for the record:
  We learn to *generate* the complete GT TSDF volume x0 (128^3, one channel) by denoising.
  Training corrupts x0 with Gaussian noise at a random timestep t, and a 3D UNet (the "main
  branch") predicts the noise that was added. Conditioning on the partial DA3 scan is done
  DiffComplete-style: a second identical encoder (the "control branch") encodes the 5-channel
  partial input and its features are ADDED into the main branch at every resolution through
  zero-initialized 1x1x1 convs. Because both volumes are voxel-aligned, addition keeps the
  conditioning spatially exact. Timestep is injected everywhere via adaLN (zero-init, starts as
  identity). Sampling is DDIM (50 steps), deterministic given a seed.

This file holds ONLY the network and the diffusion math (schedule, q_sample, loss, DDIM). The
training loop, data, and verification gates live in train_diffusion.py.
"""
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

OBSERVED, FREE, UNKNOWN = 0, 1, 2


# ----------------------------------------------------------------------------- time embedding
def timestep_embedding(t, dim):
    """Standard sinusoidal embedding of an integer timestep (ADM / DDPM)."""
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
    a = t.float()[:, None] * freqs[None]
    emb = torch.cat([torch.cos(a), torch.sin(a)], -1)
    if dim % 2:
        emb = F.pad(emb, (0, 1))
    return emb


# ----------------------------------------------------------------------------- building blocks
class ResBlock3D(nn.Module):
    """Two conv3x3x3 with GroupNorm+SiLU, plus a skip. Timestep enters via adaLN: GroupNorm is
    made affine-free and we apply (1+scale)*h + shift with scale/shift predicted from the time
    vector (zero-init -> starts as the identity)."""
    def __init__(self, cin, cout, t_dim):
        super().__init__()
        self.n1 = nn.GroupNorm(8, cin, affine=False)
        self.c1 = nn.Conv3d(cin, cout, 3, padding=1)
        self.n2 = nn.GroupNorm(8, cout, affine=False)
        self.c2 = nn.Conv3d(cout, cout, 3, padding=1)
        self.emb = nn.Linear(t_dim, 2 * cin + 2 * cout)          # adaLN params for both norms
        nn.init.zeros_(self.emb.weight); nn.init.zeros_(self.emb.bias)
        self.skip = nn.Conv3d(cin, cout, 1) if cin != cout else nn.Identity()
        self.cin, self.cout = cin, cout

    def forward(self, x, t):
        s1, b1, s2, b2 = torch.split(self.emb(t), [self.cin, self.cin, self.cout, self.cout], -1)
        d = lambda v: v[:, :, None, None, None]
        h = self.n1(x) * (1 + d(s1)) + d(b1)
        h = self.c1(F.silu(h))
        h = self.n2(h) * (1 + d(s2)) + d(b2)
        h = self.c2(F.silu(h))
        return h + self.skip(x)


class Attention3D(nn.Module):
    """Multi-head self-attention over the flattened spatial volume. Used only at 16^3 and 8^3,
    where the token count (4096 / 512) is affordable in 3D."""
    def __init__(self, ch, heads=4):
        super().__init__()
        self.heads = heads
        self.norm = nn.GroupNorm(8, ch)
        self.qkv = nn.Conv3d(ch, ch * 3, 1)
        self.proj = nn.Conv3d(ch, ch, 1)
        nn.init.zeros_(self.proj.weight); nn.init.zeros_(self.proj.bias)  # start as identity

    def forward(self, x):
        B, C, D, H, W = x.shape
        qkv = self.qkv(self.norm(x)).reshape(B, 3, self.heads, C // self.heads, D * H * W)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]                 # [B, heads, c, N]
        out = F.scaled_dot_product_attention(q.transpose(-1, -2), k.transpose(-1, -2),
                                             v.transpose(-1, -2))  # [B,heads,N,c]
        out = out.transpose(-1, -2).reshape(B, C, D, H, W)
        return x + self.proj(out)


class Down(nn.Module):
    def __init__(self, ch):
        super().__init__(); self.op = nn.Conv3d(ch, ch, 3, stride=2, padding=1)
    def forward(self, x): return self.op(x)


class Up(nn.Module):
    def __init__(self, ch_in, ch_out):
        super().__init__(); self.op = nn.ConvTranspose3d(ch_in, ch_out, 2, stride=2)
    def forward(self, x): return self.op(x)


# ----------------------------------------------------------------------------- encoder tower
class Encoder(nn.Module):
    """Shared structure for both branches: stem -> [2 resblocks (+attn) then downsample] x levels
    -> middle (resblock, attn, resblock). Returns the bottleneck plus the per-level skip features
    (which the control branch hands to the main branch, and the main branch also uses as U-Net
    skips)."""
    def __init__(self, in_ch, chs, t_dim, attn_res, res0=128, use_ckpt=True):
        super().__init__()
        self.use_ckpt = use_ckpt
        self.stem = nn.Conv3d(in_ch, chs[0], 3, padding=1)
        self.levels = nn.ModuleList(); self.downs = nn.ModuleList()
        res = res0
        for i in range(len(chs) - 1):
            blocks = nn.ModuleList([ResBlock3D(chs[i], chs[i], t_dim),
                                    ResBlock3D(chs[i], chs[i], t_dim)])
            attn = Attention3D(chs[i]) if res in attn_res else None
            self.levels.append(nn.ModuleList([blocks, attn]))
            # downsample also changes channels chs[i] -> chs[i+1] (1x1 conv then stride-2 conv)
            self.downs.append(nn.Sequential(nn.Conv3d(chs[i], chs[i + 1], 1), Down(chs[i + 1])))
            res //= 2
        self.mid = nn.ModuleList([ResBlock3D(chs[-1], chs[-1], t_dim),
                                  Attention3D(chs[-1]),
                                  ResBlock3D(chs[-1], chs[-1], t_dim)])

    def _rb(self, block, x, t):
        if self.use_ckpt and x.requires_grad:
            return checkpoint(block, x, t, use_reentrant=False)
        return block(x, t)

    def forward(self, x, t):
        x = self.stem(x)
        skips = []
        for (blocks, attn), down in zip(self.levels, self.downs):
            for b in blocks:
                x = self._rb(b, x, t)
            if attn is not None:
                x = attn(x)
            skips.append(x)                                      # pre-downsample feature = skip
            x = down(x)
        x = self._rb(self.mid[0], x, t); x = self.mid[1](x); x = self._rb(self.mid[2], x, t)
        return x, skips


# ----------------------------------------------------------------------------- full model
class DiffCompleteUNet(nn.Module):
    """Main branch (full UNet, denoises x_t) + control branch (encoder only, encodes the partial
    scan). Control features are added into the main branch at every resolution through zero-init
    1x1x1 convs, so training starts unconditioned and learns to use the condition gradually."""
    def __init__(self, chs=(32, 64, 128, 256, 256), attn_res=(16, 8), cond_ch=5,
                 t_dim=512, use_ckpt=True):
        super().__init__()
        self.chs = list(chs)
        self.time_mlp = nn.Sequential(nn.Linear(128, t_dim), nn.SiLU(), nn.Linear(t_dim, t_dim))
        # main branch encodes the 1-channel noisy TSDF; control encodes the 5-channel condition
        self.main_enc = Encoder(1, chs, t_dim, set(attn_res), use_ckpt=use_ckpt)
        self.ctrl_enc = Encoder(cond_ch, chs, t_dim, set(attn_res), use_ckpt=use_ckpt)
        # zero-init connections: bottleneck + one per skip level
        self.zc_mid = self._zero_conv(chs[-1])
        self.zc_skip = nn.ModuleList([self._zero_conv(c) for c in chs[:-1]])
        # decoder (main branch only)
        self.ups = nn.ModuleList(); self.dec = nn.ModuleList(); self.dec_attn = nn.ModuleList()
        res = 8
        for i in range(len(chs) - 1, 0, -1):
            self.ups.append(Up(chs[i], chs[i - 1]))
            self.dec.append(nn.ModuleList([ResBlock3D(chs[i - 1] * 2, chs[i - 1], t_dim),
                                           ResBlock3D(chs[i - 1], chs[i - 1], t_dim)]))
            res *= 2
            self.dec_attn.append(Attention3D(chs[i - 1]) if res in set(attn_res) else None)
        self.out_norm = nn.GroupNorm(8, chs[0])
        self.out_conv = nn.Conv3d(chs[0], 1, 1)
        nn.init.zeros_(self.out_conv.weight); nn.init.zeros_(self.out_conv.bias)  # zero-init head
        self.use_ckpt = use_ckpt

    @staticmethod
    def _zero_conv(ch):
        c = nn.Conv3d(ch, ch, 1); nn.init.zeros_(c.weight); nn.init.zeros_(c.bias); return c

    def _rb(self, block, x, t):
        if self.use_ckpt and x.requires_grad:
            return checkpoint(block, x, t, use_reentrant=False)
        return block(x, t)

    def forward(self, x_t, cond, t, cond_drop=None):
        """x_t: [B,1,128^3] noisy TSDF. cond: [B,5,128^3] partial scan. t: [B] int timesteps.
        cond_drop: optional [B] bool, zero the control contribution (classifier-free guidance)."""
        temb = self.time_mlp(timestep_embedding(t, 128))
        # control branch encodes the condition -> bottleneck + per-level features
        cmid, cskips = self.ctrl_enc(cond, temb)
        if cond_drop is not None:
            g = (~cond_drop).float()[:, None, None, None, None]
            cmid = cmid * g; cskips = [s * g for s in cskips]
        # main branch encoder, adding control features in
        x = self.main_enc.stem(x_t)
        skips = []
        for (blocks, attn), down, cs, zc in zip(self.main_enc.levels, self.main_enc.downs,
                                                cskips, self.zc_skip):
            for b in blocks:
                x = self._rb(b, x, temb)
            if attn is not None:
                x = attn(x)
            x = x + zc(cs)                                       # inject control at this resolution
            skips.append(x)
            x = down(x)
        x = self._rb(self.main_enc.mid[0], x, temb); x = self.main_enc.mid[1](x)
        x = self._rb(self.main_enc.mid[2], x, temb)
        x = x + self.zc_mid(cmid)                                # inject control at bottleneck
        # main branch decoder
        for up, blocks, attn, s in zip(self.ups, self.dec, self.dec_attn, reversed(skips)):
            x = up(x); x = torch.cat([x, s], 1)
            for b in blocks:
                x = self._rb(b, x, temb)
            if attn is not None:
                x = attn(x)
        return self.out_conv(F.silu(self.out_norm(x)))


# ----------------------------------------------------------------------------- diffusion process
class Diffusion:
    """Cosine-schedule DDPM (T=1000), epsilon-prediction, with a masked surface-weighted L1 loss
    and a DDIM sampler. Kept as plain tensors so it is easy to read and to move to any device."""
    def __init__(self, T=1000, device="cuda"):
        self.T = T
        # cosine schedule (Nichol & Dhariwal): alpha_bar(t) = cos^2(((t/T+s)/(1+s)) * pi/2)
        s = 0.008
        f = torch.cos(((torch.arange(T + 1) / T + s) / (1 + s)) * math.pi / 2) ** 2
        ab = f / f[0]
        betas = (1 - ab[1:] / ab[:-1]).clamp(1e-4, 0.999)
        self.betas = betas.to(device)
        self.ab = torch.cumprod(1 - self.betas, 0)               # alpha_bar_t, [T]
        self.sqrt_ab = self.ab.sqrt()
        self.sqrt_1mab = (1 - self.ab).sqrt()

    def q_sample(self, x0, t, noise):
        """Forward diffusion: x_t = sqrt(ab)*x0 + sqrt(1-ab)*noise."""
        return self.sqrt_ab[t][:, None, None, None, None] * x0 + \
               self.sqrt_1mab[t][:, None, None, None, None] * noise

    def p_losses(self, model, x0, cond, mask, near_surf, cond_drop=None, surf_w=5.0,
                 frontier_dilate=0):
        """Sample t, add noise, predict it, and score with masked surface-weighted L1 on the noise.
        `near_surf` is a precomputed bool mask of voxels near the true surface (from the clean
        distance field, never from x_t) — passing it in keeps the loss agnostic to whether x0 is a
        signed TSDF or an unsigned TUDF.

        `frontier_dilate` (>0, in voxels) restricts the scored region so the loss ignores the deep
        empty far-field (prone to hallucination) WITHOUT dropping real occluded geometry. It keeps an
        unknown voxel if it is near ANY TRUE SURFACE (so deep occluded structure -- table legs,
        undersides, cabinet backs -- is fully supervised; these are GT surface even if far from any
        observation) OR near an OBSERVED surface (the thin empty band that teaches "keep near-empty
        space empty"). Only voxels far from both are excluded. 0 = score all unknown.
        (The earlier version anchored ONLY to observed surface and silently dropped ~36% of real
        occluded targets -- e.g. the middle of a tall table's legs; this fixes that.)"""
        B = x0.shape[0]
        t = torch.randint(0, self.T, (B,), device=x0.device)
        noise = torch.randn_like(x0)
        x_t = self.q_sample(x0, t, noise)
        pred = model(x_t, cond, t, cond_drop=cond_drop)
        m = (mask == UNKNOWN).float()
        if frontier_dilate > 0:
            k = int(frontier_dilate)
            near_obs = F.max_pool3d((mask == OBSERVED).float(), 2 * k + 1, stride=1, padding=k) > 0
            keep = near_surf | near_obs                            # near TRUE surface OR near observed
            m = m * keep.float()
        w = torch.where(near_surf, torch.full_like(x0, surf_w), torch.ones_like(x0)) * m
        return (w * (pred - noise).abs()).sum() / w.sum().clamp_min(1.0)

    @torch.no_grad()
    def ddim_sample(self, model, cond, mask=None, x_known=None, steps=50, eta=0.0,
                    guidance=1.0, seed=0, replace=False):
        """DDIM sampling. Optional RePaint-style known-region replacement keeps observed/free
        voxels pinned to the (noised) partial TSDF at every step."""
        dev = cond.device; B = cond.shape[0]
        g = torch.Generator(dev).manual_seed(seed)
        x = torch.randn(B, 1, *cond.shape[2:], device=dev, generator=g)
        ts = torch.linspace(self.T - 1, 0, steps, device=dev).long()
        keep = None
        if replace and mask is not None:
            keep = ((mask == OBSERVED) | (mask == FREE)).float()
        for i, t in enumerate(ts):
            tb = t.repeat(B)
            eps = model(x, cond, tb)
            if guidance != 1.0:
                eps_u = model(x, cond, tb, cond_drop=torch.ones(B, dtype=torch.bool, device=dev))
                eps = eps_u + guidance * (eps - eps_u)
            ab_t = self.ab[t]
            x0 = (x - (1 - ab_t).sqrt() * eps) / ab_t.sqrt()
            x0 = x0.clamp(-1, 1)
            if i == len(ts) - 1:
                x = x0; break
            ab_p = self.ab[ts[i + 1]]
            sigma = eta * ((1 - ab_p) / (1 - ab_t) * (1 - ab_t / ab_p)).sqrt()
            x = ab_p.sqrt() * x0 + (1 - ab_p - sigma ** 2).sqrt() * eps
            if sigma > 0:
                x = x + sigma * torch.randn(x.shape, device=dev, generator=g)
            if keep is not None and x_known is not None:
                noised = self.sqrt_ab[ts[i + 1]] * x_known + self.sqrt_1mab[ts[i + 1]] * \
                         torch.randn(x.shape, device=dev, generator=g)
                x = keep * noised + (1 - keep) * x
        return x


class EMA:
    """Exponential moving average of model weights. Critical for diffusion sample quality —
    always evaluate with these, not the raw weights."""
    def __init__(self, model, decay=0.9999):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model):
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1 - self.decay)
            else:
                self.shadow[k].copy_(v)

    def copy_to(self, model):
        model.load_state_dict(self.shadow, strict=True)


if __name__ == "__main__":
    # quick shape/param sanity on CPU-ish (tiny grid to keep it fast)
    m = DiffCompleteUNet(use_ckpt=False)
    n = sum(p.numel() for p in m.parameters()) / 1e6
    print(f"params = {n:.1f}M")
    x = torch.randn(1, 1, 32, 32, 32); c = torch.randn(1, 5, 32, 32, 32); t = torch.tensor([10])
    print("out:", m(x, c, t).shape)
