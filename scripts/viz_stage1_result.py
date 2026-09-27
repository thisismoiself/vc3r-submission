#!/usr/bin/env python3
"""Visualize the Stage-1 overfit COMPLETION: run the trained UNet on the crop, extract the
surface it generated in the occluded (unknown) region, and compare to GT and the partial input.

Outputs (world frame):
  observed_surface.ply       blue   -- the visible input surface (has holes where occluded)
  gt_unknown_surface.ply     green  -- the occluded GT surface (what should be filled)
  pred_unknown_surface.ply   orange -- what the UNet GENERATED in the occluded region
  merged_output.ply          -- observed(blue) + generated(orange) = the completed cloud
  slices.png                 -- partial / GT / predicted TSDF + error, per z-slice
"""
import sys
from pathlib import Path
import numpy as np, torch, open3d as o3d
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, "/usr/prakt/s0016/vc3r/scripts")
import stage1_unet as S

GRID = "/usr/prakt/s0016/vc3r/outputs/tsdf_pipeline/office4_win0/grid.npz"
CKPT = "/usr/prakt/s0016/vc3r/outputs/consecutive_windows/stage1_overfit.pt"
OUT = Path("/usr/prakt/s0016/vc3r/outputs/tsdf_pipeline/office4_win0")
DEV = S.DEV
OBSERVED, UNKNOWN = S.OBSERVED, S.UNKNOWN


def main():
    gt_np, pt_np, mask_np, voxel = S.load_grid(GRID)
    d = np.load(GRID); center = d["center"]; band = 0.10
    N = gt_np.shape[0]
    inp = S.build_input(gt_np, pt_np, mask_np, band).unsqueeze(0).to(DEV)
    net = S.OcclusionUNet(in_ch=5, base=16, band=band).to(DEV)
    net.load_state_dict(torch.load(CKPT, map_location=DEV, weights_only=False)["model"]); net.eval()
    with torch.no_grad():
        pred = net(inp)[0, 0].cpu().numpy()

    # voxel-centre world coords
    lin = (np.arange(N) + 0.5) * voxel - N * voxel / 2
    gx, gy, gz = np.meshgrid(lin, lin, lin, indexing="ij")
    vox = np.stack([gx, gy, gz], -1) + center

    surf = voxel * 1.0                       # |tsdf| < 1 voxel = thin surface shell
    m_obs = mask_np == OBSERVED
    m_unk = mask_np == UNKNOWN
    sel_obs = (np.abs(pt_np) < surf) & m_obs
    sel_gt_unk = (np.abs(gt_np) < surf) & m_unk
    sel_pred_unk = (np.abs(pred) < surf) & m_unk

    def ply(sel, path, color):
        p = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(vox[sel]))
        p.colors = o3d.utility.Vector3dVector(np.tile(color, (int(sel.sum()), 1)))
        o3d.io.write_point_cloud(str(path), p)
        return int(sel.sum())

    n_obs = ply(sel_obs, OUT / "observed_surface.ply", [0.2, 0.55, 1.0])
    n_gt = ply(sel_gt_unk, OUT / "gt_unknown_surface.ply", [0.2, 0.85, 0.3])
    n_pr = ply(sel_pred_unk, OUT / "pred_unknown_surface.ply", [1.0, 0.5, 0.1])
    # merged completed output: observed + generated
    both = sel_obs | sel_pred_unk
    cols = np.zeros((int(both.sum()), 3)); pts = vox[both]
    obs_of_both = sel_obs[both]
    cols[obs_of_both] = [0.2, 0.55, 1.0]; cols[~obs_of_both] = [1.0, 0.5, 0.1]
    mp = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
    mp.colors = o3d.utility.Vector3dVector(cols); o3d.io.write_point_cloud(str(OUT / "merged_output.ply"), mp)

    # accuracy of the generated surface vs GT surface, in the occluded region (voxel-level)
    from scipy.spatial import cKDTree
    if n_pr and n_gt:
        gt_pts = vox[sel_gt_unk]; pr_pts = vox[sel_pred_unk]
        d_pg = cKDTree(gt_pts).query(pr_pts)[0]; d_gp = cKDTree(pr_pts).query(gt_pts)[0]
        prec = float((d_pg < 0.02).mean()); rec = float((d_gp < 0.02).mean())
        f2 = 2 * prec * rec / (prec + rec + 1e-9)
    else:
        prec = rec = f2 = 0.0

    # slices
    fig, ax = plt.subplots(4, 3, figsize=(11, 14))
    ks = [N // 4, N // 2, 3 * N // 4]
    err = np.abs(pred - gt_np) * (mask_np == UNKNOWN)
    for c, k in enumerate(ks):
        ax[0, c].imshow(pt_np[:, :, k], cmap="coolwarm", vmin=-band, vmax=band); ax[0, c].set_title(f"partial TSDF z{k}")
        ax[1, c].imshow(gt_np[:, :, k], cmap="coolwarm", vmin=-band, vmax=band); ax[1, c].set_title(f"GT TSDF z{k}")
        ax[2, c].imshow(pred[:, :, k], cmap="coolwarm", vmin=-band, vmax=band); ax[2, c].set_title(f"PREDICTED TSDF z{k}")
        ax[3, c].imshow(err[:, :, k], cmap="magma", vmin=0, vmax=0.03); ax[3, c].set_title(f"|pred-GT| on unknown z{k}")
    for a in ax.ravel(): a.axis("off")
    plt.tight_layout(); plt.savefig(OUT / "result_slices.png", dpi=110); plt.close()

    print(f"observed_surface={n_obs:,}  gt_unknown={n_gt:,}  pred_unknown={n_pr:,}")
    print(f"generated-vs-GT (occluded region): F@2cm={f2:.3f} prec={prec:.3f} rec={rec:.3f}")
    import json
    (OUT / "result_stats.json").write_text(json.dumps(
        {"observed": n_obs, "gt_unknown": n_gt, "pred_unknown": n_pr,
         "gen_f2": f2, "gen_prec": prec, "gen_rec": rec}, indent=2))
    print(f"[out] {OUT}")


if __name__ == "__main__":
    main()
