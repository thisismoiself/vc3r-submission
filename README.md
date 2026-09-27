# VC3R Office4 Evaluation

This repository provides the VC3R implementation and a reproducible evaluation
of the four-dataset model on the Replica `office4` scene.

## 1. Clone the repository

```bash
git clone --recurse-submodules --branch prep-main <REPOSITORY_URL> vc3r
cd vc3r
```

For an existing clone, initialize the pinned DA3 and NOVA3R dependencies with:

```bash
git submodule update --init --recursive
```

## 2. Create the evaluation environment

Install Miniconda, then run:

```bash
scripts/setup_eval_env.sh --conda /path/to/miniconda3/bin/conda
```

This creates the environment at `.conda/envs/vc3r-eval`.

## 3. Acquire the model artifacts

Download the pinned DA3 snapshot:

```bash
.conda/envs/vc3r-eval/bin/python scripts/download_da3.py
```

Provide the remaining artifacts using this layout:

```text
checkpoints/
├── replica-nrgbd-7scenes-scannetpp.pt
├── da3/DA3-LARGE-1.1/
│   ├── config.json
│   └── model.safetensors
└── nova3r/scene_ae/
    ├── checkpoint-last.pth
    └── .hydra/config.yaml
```

You can obtain the nova3r checkpoint by following the official download instructions from the NOVA3R repository.

## 4. Prepare Replica

The evaluator expects:

```text
REPLICA_ROOT/
├── cam_params.json
├── office4_mesh.ply
└── office4/
    ├── traj.txt
    └── results/
        ├── frame000000.jpg
        ├── depth000000.png
        └── ...
```

On the project cluster, the shared dataset can be linked with:

```bash
mkdir -p datasets
ln -s /storage/group/cvpr/chwe/da3_nova3r/replica datasets/replica
```

## 5. Run the full Office4 evaluation

Local execution:

```bash
scripts/reproduce_office4.sh \
  checkpoints/replica-nrgbd-7scenes-scannetpp.pt \
  datasets/replica \
  --room office4 \
  --complete-target \
  --da3pose-tokens \
  --n-frames 8 \
  --stride 10 \
  --fm-sampling midpoint \
  --fm-step-size 0.04 \
  --num-queries 50000 \
  --seed 42 \
  --da3-model checkpoints/da3/DA3-LARGE-1.1 \
  --offline \
  --out-tag four_dataset_gtfree
```

SLURM execution uses the same evaluator arguments:

```bash
sbatch scripts/slurm/eval_office4.sbatch \
  checkpoints/replica-nrgbd-7scenes-scannetpp.pt \
  --room office4 \
  --complete-target \
  --da3pose-tokens \
  --n-frames 8 \
  --stride 10 \
  --fm-sampling midpoint \
  --fm-step-size 0.04 \
  --num-queries 50000 \
  --seed 42 \
  --da3-model checkpoints/da3/DA3-LARGE-1.1 \
  --offline \
  --out-tag four_dataset_gtfree
```

Omitting `--max-windows` evaluates all 25 Office4 windows. Results are written
to `outputs/replica/stitch_office4_four_dataset_gtfree/` and include the
stitched point clouds, per-window data, and `metrics.json`.

The checkpoints are interchangeable. Our report further explains the differences in training among them:
- `replica-gtfree.pt`: Trained through token-matched MSE + velocity loss only on Replica (synthetic)
- `replica-nrgbd-gtfree.pt`: Trained through token-matched MSE + velocity loss on Replica (synthetic) and NeuralRGBD (synthetic)
- `replica-nrgbd-7scenes-scannetpp.pt`: Trained on only token-matched loss on Replica (synthetic), NeuralRGBD (synthetic), 7scenes (real-world) and ScanNet++ (real-world)


## Different datasets can be evaluated similary after preparing and providing them in the same folder structure

## Training
The provided model checkpoints have been trained with different scripts, which can be found here:
- `scripts/train_online_var_hungarian.py`
- `scripts/cache_scannetpp.sh`
- `scripts/cache_online_hungarian_zstar_windows.py`

The caching scripts pre-cache tokens to accelerate training over multiple runs. 
