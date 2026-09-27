#!/usr/bin/env python3
"""Detached reporter: waits for the point-flow stitch and the NRGBD SLURM run to finish,
then writes clean headline numbers to RESULTS_WHILE_AWAY.md. Survives SSH disconnect
(launched via setsid+nohup), so results are ready on the shared FS regardless of whether
the Claude Code session is alive."""
import json, subprocess, time
from pathlib import Path
from datetime import datetime

REPO = Path("/usr/prakt/s0016/vc3r")
OUT = REPO / "experiments/overfit_8frames/RESULTS_WHILE_AWAY.md"
PF_METRICS = REPO / "outputs/replica/stitch_office4_pf_corrected_midpoint/metrics.json"
PF_BASELINE = REPO / "outputs/replica/stitch_office4_fullrec_nf16_vel03_midpoint/metrics.json"
NRGBD_METRICS = REPO / "outputs/replica/stitch_office4_fullrec_nrgbd_midpoint/metrics.json"
NRGBD_JOB = "1629710"

BASE = "baseline (Replica-only, no corrector): whole chamfer 0.0567 | FURN F@2 0.3709 F@5 0.667"
ORAC = "oracle ceiling:                          whole chamfer 0.0440 | FURN F@2 0.5510 F@5 0.807"


def load(p):
    try:
        return json.load(open(p))
    except Exception:
        return None


def fmt(tag, m):
    if not m:
        return f"{tag}: (metrics not found yet)"
    p, pf = m["pred"], m["pred_furniture"]
    o, of = m.get("oracle", {}), m.get("oracle_furniture", {})
    s = (f"{tag}:\n"
         f"  PRED   : whole chamfer {p['chamfer_m']:.4f}  F@5 {p['F@5cm']:.3f}  "
         f"| FURN F@2 {pf['F@2cm']:.4f}  F@5 {pf['F@5cm']:.4f}\n")
    if of:
        s += (f"  ORACLE : whole chamfer {o['chamfer_m']:.4f}  F@5 {o['F@5cm']:.3f}  "
              f"| FURN F@2 {of['F@2cm']:.4f}  F@5 {of['F@5cm']:.4f}\n")
    return s


def controlled_delta(base, corr):
    if not base or not corr:
        return "CONTROLLED DELTA: (waiting for baseline run to finish)"
    bf, cf = base["pred_furniture"], corr["pred_furniture"]
    bw, cw = base["pred"], corr["pred"]
    return ("CONTROLLED DELTA (corrected - baseline, same seed):\n"
            f"  whole chamfer {cw['chamfer_m']-bw['chamfer_m']:+.4f}  "
            f"whole F@5 {cw['F@5cm']-bw['F@5cm']:+.3f}  "
            f"| FURN F@2 {cf['F@2cm']-bf['F@2cm']:+.4f}  FURN F@5 {cf['F@5cm']-bf['F@5cm']:+.4f}\n"
            "  (FURN F@2 > 0 => corrector helps furniture detail; < 0 => bulk-smoother hurts it)")


def squeue_has(job):
    try:
        r = subprocess.run(["squeue", "-j", job, "-h"], capture_output=True, text=True, timeout=30)
        return bool(r.stdout.strip())
    except Exception:
        return False


def write(pf_done, nrgbd_done):
    lines = [
        "# Results while away", "",
        f"_updated {datetime.now():%Y-%m-%d %H:%M:%S}_", "",
        "## Reference", f"- {BASE}", f"- {ORAC}", "",
        "## 1. Point-flow corrector on office4 (your point-space FM idea)",
        "```", fmt("BASELINE (this run, no corrector, seed 42)", load(PF_BASELINE)),
        "", fmt("CORRECTED (this run, seed 42)", load(PF_METRICS)),
        "```",
        controlled_delta(load(PF_BASELINE), load(PF_METRICS)),
        ("STATUS: corrected DONE" if pf_done else "STATUS: corrected still running"),
        ("  baseline DONE" if PF_BASELINE.exists() else "  baseline still running"), "",
        "Read: the BASELINE above is the SAME run/seed with no corrector -> apples-to-apples,",
        "free of the ~0.006 cross-run furniture noise. Whole-scene already improved (chamfer,",
        "F@5); the controlled furniture delta below is the real verdict.", "",
        "## 2. NRGBD data-scaling on office4 (Replica-7 + NRGBD-9, val=office4)",
        "```", fmt("NRGBD-DATA", load(NRGBD_METRICS)),
        "```",
        ("STATUS: DONE" if nrgbd_done else "STATUS: SLURM job still running"), "",
        "Read: FURN F@2 vs baseline 0.3709 (did +9 scenes of data help office4 detail?).", "",
        "## Raw logs",
        "- point-flow stitch: experiments/overfit_8frames/eval_pf_corrected_stitch.log",
        "- point-flow train:  experiments/overfit_8frames/point_flow_full.log",
        "- NRGBD train:       experiments/overfit_8frames/fullrec_nrgbd.log",
        "- NRGBD stitch:      experiments/overfit_8frames/eval_fullrec_nrgbd_stitch.log",
    ]
    OUT.write_text("\n".join(lines))


deadline = time.time() + 8 * 3600
while time.time() < deadline:
    pf_done = PF_METRICS.exists() and PF_BASELINE.exists()
    nrgbd_done = (not squeue_has(NRGBD_JOB)) and NRGBD_METRICS.exists()
    write(pf_done, nrgbd_done)
    if pf_done and nrgbd_done:
        break
    time.sleep(120)
write(PF_METRICS.exists(), NRGBD_METRICS.exists())
