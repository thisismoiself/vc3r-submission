#!/usr/bin/env python3
"""Submit final-checkpoint office4 evaluations one at a time and report results."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

REPO = Path(__file__).resolve().parents[1]
REPORT = REPO / 'checkpoints/final/office4_results.md'
RUN_ID = time.strftime('%Y%m%d_%H%M%S')


def write_report(rows):
    lines = ['# Final checkpoints: office4 evaluation', '',
             'Sequential SLURM runs; each job finishes before the next is submitted.', '',
             'Protocol: all 25 windows, 8 frames/window, stride 10, 50,000 decoded points/window, '
             'complete targets, midpoint ODE (step 0.04), decode seed 42. '
             'NOVA3R: `checkpoints/nova3r/scene_ae/checkpoint-last.pth`. '
             'Distances are cm; F-scores, precision and recall are percentages.', '',
             'Token poses match training: `da3pose` uses DA3 predictions; `fullrec` uses GT. '
             'All runs still use GT for placement and scale. The point-flow checkpoint is a '
             'corrector, evaluated with `fullrec_nf16_vel03_complete_best.pt` and 6 correction steps. '
             'Mesh sampling is not globally seeded by the evaluation script, so small run-to-run '
             'differences (including oracle differences) are expected. '
             'The nic/replica.pt, nic/replica-nrgbd.pt and nic/flow-matching.pt files are '
             'byte-identical copies of the corresponding fullrec/point-flow checkpoints, rerun separately. '
             'The nic four-dataset adapter is evaluated with GT poses and complete targets; '
             'its metadata does not confirm these settings.', '',
             '| Checkpoint | Token poses | Job | Status | Elapsed incl. queue/cleanup (s) | Results |',
             '|---|---|---|---|---:|---|']
    for r in rows:
        link = f"[metrics]({r['output']}/metrics.json)" if 'metrics' in r else f"[log]({r['log']})"
        lines.append(f"| {r['name']} | {r['pose']} | {r.get('job', '—')} | {r['status']} | {r.get('seconds', '—')} | {link} |")
    for key, title, thresholds in [('pred', 'Adapter reconstruction', [1, 5]),
                                   ('pred_furniture', 'Furniture region', [2, 5]),
                                   ('oracle', 'NOVA3R oracle', [1, 5]),
                                   ('oracle_furniture', 'Oracle furniture region', [2, 5]),
                                   ('input', 'Input geometry sanity check', [1, 5])]:
        fields = [('accuracy_m', 'Accuracy ↓'), ('completeness_m', 'Completeness ↓'),
                  ('chamfer_m', 'Chamfer ↓'), ('acc_median_m', 'Median accuracy ↓'),
                  ('comp_median_m', 'Median completeness ↓')]
        for t in thresholds:
            fields += [(f'F@{t}cm', f'F@{t}cm ↑'), (f'prec@{t}cm', f'Precision@{t}cm ↑'),
                       (f'recall@{t}cm', f'Recall@{t}cm ↑')]
        lines += ['', f'## {title}', '', '| Checkpoint | ' + ' | '.join(label for _, label in fields) + ' |',
                  '|---|' + '---:|' * len(fields)]
        for r in rows:
            if 'metrics' in r:
                m = r['metrics'][key]
                lines.append('| ' + r['name'] + ' | ' + ' | '.join(f'{100*m[k]:.2f}' for k, _ in fields) + ' |')
    REPORT.write_text('\n'.join(lines) + '\n')


def main():
    os.chdir(REPO)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint-dir', type=Path, default=REPO / 'checkpoints/final')
    parser.add_argument('--append', action='store_true', help='Preserve existing report rows')
    args = parser.parse_args()
    checkpoint_dir = args.checkpoint_dir.resolve()
    paths = sorted(checkpoint_dir.glob('*.pt'))
    if not paths:
        raise SystemExit(f'No checkpoints in {checkpoint_dir}')
    if checkpoint_dir.name == 'nic':
        paths.sort(key=lambda p: ['replica.pt', 'replica-nrgbd.pt',
                                 'replica-nrgbd-7scenes-scannetpp.pt', 'flow-matching.pt'].index(p.name))
    rows = []
    if args.append and REPORT.exists():
        for line in REPORT.read_text().splitlines():
            if not line.startswith('| ') or '[metrics](' not in line:
                continue
            name, pose, job, status, seconds, link = [x.strip() for x in line.strip('|').split('|')]
            metrics_path = Path(link.split('[metrics](', 1)[1].split(')', 1)[0])
            rows.append(dict(name=name, pose=pose, job=job, status=status,
                             seconds=int(seconds), output=str(metrics_path.parent),
                             metrics=json.loads(metrics_path.read_text())))
    new_rows = []
    for p in paths:
        label = str(p.relative_to(REPO / 'checkpoints/final'))
        tag = f'final_{RUN_ID}_{label.replace(chr(47), chr(95))[:-3]}'
        new_rows.append(dict(name=label, pose='DA3 predicted' if p.name.startswith('da3pose') else 'GT',
                         status='Pending', output=str(REPO / f'outputs/replica/stitch_office4_{tag}'),
                         log=str(REPO / f'logs/stitch_gtfree/{tag}.submission.log'), tag=tag))
    rows.extend(new_rows)
    write_report(rows)
    for p, r in zip(paths, new_rows):
        cmd = ['sbatch', '--parsable', '--wait', 'scripts/slurm/stitch_gtfree.sbatch', '--ckpt', str(p), '--out-tag', r['tag']]
        if p.name in ('point_flow_complete.pt', 'flow-matching.pt'):
            base = p.with_name('replica.pt' if p.name == 'flow-matching.pt' else 'fullrec_nf16_vel03_complete_best.pt')
            cmd[cmd.index('--ckpt') + 1] = str(base)
            cmd += ['--point-corrector', str(p), '--corrector-steps', '6']
        env = dict(os.environ, POSE_SOURCE='da3pred' if r['pose'] == 'DA3 predicted' else 'gt')
        start = time.monotonic()
        print(f'Submitting {p.name}', flush=True)
        with open(r['log'], 'w') as log:
            proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=log, text=True)
            job = proc.stdout.readline().strip().split(';')[0]
            r.update(job=job, status='Queued/running', log=str(REPO / f'logs/stitch_gtfree/{job}.out'))
            write_report(rows)
            print(f'Job {job}: {p.name}', flush=True)
            code = proc.wait()
        r['seconds'] = round(time.monotonic() - start)
        metrics = Path(r['output']) / 'metrics.json'
        r['status'] = 'Completed' if code == 0 and metrics.exists() else f'Failed (exit {code})'
        if metrics.exists() and code == 0:
            r['metrics'] = json.loads(metrics.read_text())
        write_report(rows)
        print(f"{r['status']}: {p.name} ({r['seconds']}s)", flush=True)
    print(f'Report: {REPORT}', flush=True)
    return int(any(r['status'] != 'Completed' for r in rows))


if __name__ == '__main__':
    raise SystemExit(main())
