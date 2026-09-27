#!/usr/bin/env python3
"""
Test whether gradients survive backprop through the NOVA3R ODE decoder.

Loads a cached z_star, creates a perturbed z_pred (requires_grad=True),
decodes with enable_grad=True, computes a Chamfer-style loss, and checks:
  - Does grad exist and is it non-zero?
  - What is the grad norm vs z_star norm?
  - How much memory and time does it cost?
  - Does a gradient step on z_pred reduce the decoded loss?
  - How do 5 ODE steps compare to 25?

Usage:
  ! python experiments/test_ode_grad.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT   = Path(__file__).resolve().parents[1]
NOVA3R_ROOT = REPO_ROOT / "nova3r"
NOVA3R_3P   = NOVA3R_ROOT / "third_party"
DA3_SRC     = REPO_ROOT / "da3" / "src"
SCRIPTS_SRC = REPO_ROOT / "scripts"
OVERFIT_SRC = REPO_ROOT / "experiments" / "overfit_8frames"

for _p in [str(NOVA3R_3P), str(NOVA3R_ROOT), str(DA3_SRC), str(SCRIPTS_SRC), str(OVERFIT_SRC)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from omegaconf import OmegaConf
from demo_nova3r import load_model as load_nova3r_model
from nova3r.inference import amp_dtype_mapping
from nova3r.models.model_wrapper import BatchModelWrapper
from nova3r.flow_matching.solver import ODESolver

NUM_DECODE  = 1024   # reduced for CPU feasibility
WINDOW_DIR  = REPO_ROOT / "scripts" / "data" / "windows" / "start_0136"


class GradEnabledWrapper(BatchModelWrapper):
    """BatchModelWrapper without @torch.no_grad() so gradients flow through the ODE."""
    def forward(self, x, t, images, encoder_data=None, **extras):
        if len(t.shape) == 0:
            B = x.shape[0]
            t = t.reshape(-1, 1).expand(B, x.shape[1])
        if encoder_data is None:
            raise ValueError("encoder_data is required.")
        output = self._model._decode(
            tokens=encoder_data['tokens'], images=images,
            query_points=x, timestep=t,
        )
        return output['pts3d_xyz']


def decode(nova_model, nova_cfg, z_tokens, pts_norm, device,
           n_steps: int, seed: int, grad: bool) -> torch.Tensor:
    torch.manual_seed(seed)
    encoder_data = {"tokens": z_tokens.to(device)}
    images  = torch.zeros(1, 1, 3, 1, 1, device=device)
    x_init  = torch.rand(1, NUM_DECODE, 3, device=device) * 2 - 1
    wrapper = GradEnabledWrapper(model=nova_model) if grad else BatchModelWrapper(model=nova_model)
    solver  = ODESolver(velocity_model=wrapper)
    step_sz = 1.0 / n_steps
    method  = nova_cfg.get("fm_sampling", "euler")
    amp_dt  = amp_dtype_mapping.get(nova_cfg.get("amp_dtype", "bf16"), torch.float32)
    T_grid  = torch.linspace(0, 1, n_steps + 1).to(device)
    use_amp = device.type != "cpu"
    with torch.amp.autocast('cuda', enabled=use_amp, dtype=amp_dt):
        sol = solver.sample(
            time_grid=T_grid, x_init=x_init, method=method,
            step_size=step_sz, return_intermediates=False,
            images=images, token_mask=None,
            encoder_data=encoder_data, pointmaps=pts_norm.to(device),
            enable_grad=grad,
        )
    return sol


def mem_mb(device) -> float:
    if device.type == "cuda":
        return torch.cuda.memory_allocated(device) / 1e6
    return 0.0


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    print("Loading NOVA3R …")
    cfg_path = OVERFIT_SRC / "config.yaml"
    nova_cfg_omg = OmegaConf.load(cfg_path)
    OmegaConf.set_struct(nova_cfg_omg, False)
    ckpt = str(NOVA3R_ROOT / "checkpoints" / "scene_ae" / "checkpoint-last.pth")
    nova_model, nova_cfg = load_nova3r_model(ckpt, str(device))
    nova_model.eval()
    for p in nova_model.parameters():
        p.requires_grad_(False)
    OmegaConf.set_struct(nova_cfg, False)

    print(f"Loading cached window from {WINDOW_DIR} …")
    z_star   = torch.load(WINDOW_DIR / "z_star_consensus.pt", weights_only=True).float()
    pts_norm = torch.load(WINDOW_DIR / "pts_norm.pt",         weights_only=True).float()
    print(f"  z_star shape: {z_star.shape}   pts_norm shape: {pts_norm.shape}\n")

    # Perturb z_star to simulate an imperfect adapter prediction
    torch.manual_seed(0)
    z_pred = (z_star + 0.1 * torch.randn_like(z_star)).requires_grad_(True)

    # ── reference: decode GT z_star (no grad) ────────────────────────────────
    with torch.no_grad():
        ref_pts = decode(nova_model, nova_cfg, z_star, pts_norm, device,
                         n_steps=5, seed=42, grad=False)
    print(f"Reference decode (z_star, 5 steps): {ref_pts.shape}")

    # ── test gradient flow ────────────────────────────────────────────────────
    for n_steps in [5, 3]:
        print(f"\n── {n_steps} ODE steps ────────────────────────────────")

        if z_pred.grad is not None:
            z_pred.grad.zero_()

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize()

        mem_before = mem_mb(device)
        t0 = time.time()

        decoded = decode(nova_model, nova_cfg, z_pred, pts_norm, device,
                         n_steps=n_steps, seed=42, grad=True)

        # Simple MSE loss against reference decoded points
        # (stand-in for Chamfer; avoids the O(N²) cost here)
        loss = ((decoded - ref_pts.detach()) ** 2).mean()
        loss.backward()

        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.time() - t0

        if device.type == "cuda":
            peak_mb = torch.cuda.max_memory_allocated(device) / 1e6
        else:
            peak_mb = float("nan")

        grad = z_pred.grad
        if grad is None:
            print("  grad: NONE — gradients did not flow back")
        else:
            gn  = grad.norm().item()
            zn  = z_star.norm().item()
            ratio = gn / (zn + 1e-8)
            print(f"  loss:           {loss.item():.6f}")
            print(f"  grad norm:      {gn:.6f}")
            print(f"  z_star norm:    {zn:.6f}")
            print(f"  ratio g/z:      {ratio:.6f}")
            print(f"  grad non-zero:  {(grad.abs() > 1e-12).float().mean().item()*100:.1f}% of elements")
            print(f"  time:           {elapsed:.2f}s")
            print(f"  peak VRAM:      {peak_mb:.0f} MB")

            # ── gradient step test: does the grad actually help? ──────────────
            print(f"\n  Gradient step test ({n_steps} steps):")
            with torch.no_grad():
                # Loss before step
                d_before = decode(nova_model, nova_cfg, z_pred.detach(), pts_norm, device,
                                  n_steps=n_steps, seed=42, grad=False)
                loss_before = ((d_before - ref_pts.detach()) ** 2).mean().item()

                # Take a gradient step
                lr = 0.1
                z_stepped = (z_pred - lr * grad).detach()

                d_after = decode(nova_model, nova_cfg, z_stepped, pts_norm, device,
                                 n_steps=n_steps, seed=42, grad=False)
                loss_after = ((d_after - ref_pts.detach()) ** 2).mean().item()

            print(f"    loss before step: {loss_before:.6f}")
            print(f"    loss after  step: {loss_after:.6f}")
            improvement = (loss_before - loss_after) / (loss_before + 1e-8) * 100
            print(f"    improvement:      {improvement:.1f}%")

        z_pred.grad = None   # reset for next iteration

    print("\nDone.")


if __name__ == "__main__":
    main()
