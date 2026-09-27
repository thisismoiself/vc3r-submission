"""Decode the PURE-FM adapter (fm_only_complete_best_last.pt) on the same office0/office4 single
windows as export_complete_windows.py, and write a *_fm.ply alongside the existing pred/oracle/gt.
Reports acc/comp vs the already-exported complete-frustum GT for a quick number."""
import sys, os, json, numpy as np, torch, trimesh
from pathlib import Path
from scipy.spatial import cKDTree
REPO = Path("/usr/prakt/s0016/vc3r"); os.chdir(REPO)
for _p in ["nova3r_lib/third_party", "nova3r_lib", "da3/src", "scripts"]:
    sys.path.insert(0, _p)
os.environ.setdefault("NOVA3R_DIR", "nova3r_lib")
from demo_nova3r import load_model as load_nova3r_model
from vc3r.alignment import DA3ToNOVA3RAlignment
from nova3r.models.model_wrapper import BatchModelWrapper
from nova3r.flow_matching.solver import ODESolver
from omegaconf import OmegaConf
DEV = torch.device("cuda")
NOVA_CKPT = REPO/"checkpoints/nova3r/scene_ae/checkpoint-last.pth"
CKPT = REPO/"outputs/consecutive_windows/fm_only_complete_best_last.pt"
CACHE = REPO/"scripts/data/fc_nf16_span24_100_l13_complete"
NQ_TOTAL, CHUNK, SEED = 200_000, 50_000, 42

nova, ncfg = load_nova3r_model(str(NOVA_CKPT), str(DEV)); nova.eval()
[p.requires_grad_(False) for p in nova.parameters()]
OmegaConf.set_struct(ncfg, False); ncfg["fm_sampling"] = "midpoint"
sd = torch.load(CKPT, map_location="cpu", weights_only=False)["state_dict"]
def acfg(sd):
    tt, hd = sd["target_queries"].shape
    return dict(source_dim=sd["source_proj.weight"].shape[1], hidden_dim=hd, target_tokens=tt,
                target_dim=sd["out_proj.weight"].shape[0],
                depth=sum(1 for k in sd if k.endswith(".query_norm.weight")), num_heads=hd//64)
adapter = DA3ToNOVA3RAlignment(**acfg(sd), drop=0.0).to(DEV); adapter.load_state_dict(sd); adapter.eval()

@torch.no_grad()
def decode_dense(tokens, pts_norm):
    step = ncfg.get("fm_step_size", 0.04); T = torch.linspace(0, 1, int(1//step)).to(DEV)
    solver = ODESolver(velocity_model=BatchModelWrapper(model=nova)); outs = []
    for ci, s in enumerate(range(0, NQ_TOTAL, CHUNK)):
        n = min(CHUNK, NQ_TOTAL - s); torch.manual_seed(SEED + ci)
        xi = torch.rand(1, n, 3, device=DEV)*2-1
        with torch.amp.autocast("cuda", enabled=False):
            sol = solver.sample(time_grid=T, x_init=xi, method="midpoint", step_size=step,
                return_intermediates=False, images=torch.zeros(1,1,3,1,1,device=DEV),
                token_mask=None, encoder_data={"tokens": tokens.to(DEV)}, pointmaps=pts_norm.to(DEV))
        outs.append((sol[-1] if isinstance(sol, list) else sol)[0].cpu().float().numpy()); torch.cuda.empty_cache()
    return np.concatenate(outs, 0)

def astensor(x): return x.float() if torch.is_tensor(x) else torch.tensor(np.array(x), dtype=torch.float32)

JOBS = [
    ("VAL office4",   CACHE/"office4/1028_1050_12f_28s", REPO/"outputs/replica/single_window_complete_val_office4"),
    ("TRAIN office0", CACHE/"office0/1085_1123_14f_41s", REPO/"outputs/replica/single_window_complete_train_office0"),
]
for tag, win, out_dir in JOBS:
    win, out_dir = Path(win), Path(out_dir)
    meta = torch.load(win/"meta.pt", weights_only=False)
    da3 = torch.load(win/"da3_tokens.pt", weights_only=False).float()
    pts_norm = torch.load(win/"pts_norm.pt", weights_only=False).float()
    nf = float(meta["norm_factor"]); room = meta["room"]; scale = nf/3.0
    with torch.no_grad(): zpred = adapter(da3.to(DEV)).cpu()
    fm = decode_dense(zpred, pts_norm) * scale
    outp = out_dir / f"{room}_{win.name}_fm.ply"
    trimesh.PointCloud(fm).export(outp)
    gt_ply = out_dir / f"{room}_{win.name}_gt_complete.ply"
    msg = f"{len(fm):,} pts"
    if gt_ply.exists():
        gt = np.asarray(trimesh.load(str(gt_ply), process=False).vertices, np.float32)
        acc = cKDTree(gt).query(fm, k=1, workers=4)[0].mean()*100
        comp = cKDTree(fm).query(gt, k=1, workers=4)[0].mean()*100
        msg += f" | FM->GT acc={acc:.2f}cm comp={comp:.2f}cm"
    print(f"[{tag}] {room}/{win.name}: {msg} -> {outp.name}", flush=True)
print("DONE")
