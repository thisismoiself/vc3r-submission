# Submission Artifact Manifest

## Status and scope

This is a provisional inventory for the Replica `office4` evaluator. It records
the artifacts currently present in the working environment and the unresolved
distribution questions. Inclusion in this manifest does not mean an artifact
will be shipped with the final submission.

All hashes were computed from the local files on 2026-09-27 using SHA-256.

## Project checkpoints

All checkpoint files below currently exist locally but are ignored by the
repository's `*.pt` rule and are not tracked by Git.

| Artifact | Bytes | SHA-256 | Purpose |
|---|---:|---|---|
| `checkpoints/final/da3pose_nf16_vel03_complete_best.pt` | 73,346,484 | `da4b4ef7f806d77c6c57fe8eac27a6f5aeb2594e694fc0ef7b26ad28c1efc13e` | Replica adapter trained/evaluated with DA3-predicted pose tokens |
| `checkpoints/final/da3pose_nrgbd_complete_holdout_bfr_best.pt` | 73,347,185 | `0e73a3210433ea841e1ee4f8402d346f7db06d04bb134edd79458dc48ecc1015` | Replica + NeuralRGBD adapter with predicted-pose tokens |
| `checkpoints/final/fullrec_nf16_vel03_complete_best.pt` | 73,346,484 | `fade999fbdfc99c26da965d078394adaffd78bb0e6d061af9a9dbaf0559e1c5c` | Replica complete-target adapter with GT-pose tokens |
| `checkpoints/final/fullrec_nrgbd_complete_holdout_bfr_best.pt` | 73,347,185 | `4fb0b5009e6505f39e64fdb23fa78c612adc618881f6b73658327d741a776989` | Replica + NeuralRGBD complete-target adapter with GT-pose tokens |
| `checkpoints/final/nic/replica-nrgbd-7scenes-scannetpp.pt` | 73,339,399 | `982f9f9225c9035bee4a9b2e81910a63d6af47206d20ee5eb72691f3484f7648` | Four-dataset adapter |
| `checkpoints/final/point_flow_complete.pt` | 17,216,432 | `d47bbc2a5b0c6b00907fd64c6c733c980973a4bad43ede8a880b448ea01cbb6f` | Optional point-flow corrector |

Duplicate convenience copies:

| Artifact | Identical to |
|---|---|
| `checkpoints/final/nic/replica.pt` | `fullrec_nf16_vel03_complete_best.pt` |
| `checkpoints/final/nic/replica-nrgbd.pt` | `fullrec_nrgbd_complete_holdout_bfr_best.pt` |
| `checkpoints/final/nic/flow-matching.pt` | `point_flow_complete.pt` |

The final branch should retain only the selected public names or provide stable
download URLs. Whether the project checkpoints may be redistributed publicly is
**TBD** and must be confirmed before submission.

## NOVA3R scene autoencoder

Both files are required together by the current checkpoint loader and are
ignored by Git.

| Artifact | Bytes | SHA-256 |
|---|---:|---|
| `checkpoints/nova3r/scene_ae/checkpoint-last.pth` | 274,260,207 | `0f84c247bfd2585965753b09d4eceb1d04701f993101bea1c53b82323545f5ee` |
| `checkpoints/nova3r/scene_ae/.hydra/config.yaml` | 1,449 | `9b20baa5ca4710de17b5cf72aed9a83f2a29b222b174f676d7555e462499f998` |

The checkpoint is an upstream NOVA3R artifact and is not currently
redistributed by this repository. The final documentation must provide its
official acquisition location and applicable terms. Do not upload it with the
submission until redistribution permission has been confirmed.

The unused local `nova3r/checkpoints/scene_n1/checkpoint-last.pth` is about
6.2 GB and is not required for `stitch_office4.py`.

## Depth Anything 3

- Model ID: `depth-anything/DA3-LARGE-1.1`
- Pinned Hugging Face revision:
  `0e109ae307c5982f319a67cf6f9f99ccdc0ec97c`
- Download the exact snapshot with `python scripts/download_da3.py`. It is
  stored by default at `checkpoints/da3/DA3-LARGE-1.1`.
- Pass that directory with `--da3-model` and add `--offline` to prohibit network
  access. Passing the model ID instead resolves only the pinned revision.
- The weights are not stored in this repository.

| Snapshot file | Bytes | SHA-256 |
|---|---:|---|
| `config.json` | 1,213 | `744dcaf53859490ed92fc6cb98d68d3daf624b8c54533aaf604bdb53f06321f5` |
| `model.safetensors` | 1,643,843,860 | `739905c423cf0d6ccaf9e61a8401d82ba1ac32d7f4d3ee6dca8f92b377633f64` |

The model card/license terms for this exact weight release and redistribution
policy are **TBD** and must be verified before submission.

## Replica office4 data

The evaluator currently used the shared local dataset at
`/storage/group/cvpr/chwe/da3_nova3r/replica`. This path is machine-specific and
must appear only as historical provenance, never as a final runtime default.

Required portable layout:

```text
REPLICA_ROOT/
  cam_params.json
  office4_mesh.ply
  office4/
    traj.txt
    results/
      frame000000.jpg
      depth000000.png
      ...
```

The current copy contains 2,000 RGB images and 2,000 depth maps. Replica data is
external and ignored by Git. The final README must link to the authorized
dataset acquisition procedure and must not redistribute the dataset unless its
license explicitly permits it. The precise citation, license text, and download
instructions are **TBD**.

## Fixed evaluation cloud

| Artifact | Bytes | SHA-256 | Git state |
|---|---:|---|---|
| `outputs/replica/gt_pointclouds/office4/office4_gt_2m.ply` | 30,000,269 | `3625cf321566796ae1aa991cf81f68156150402f508d733a6ff9264fb0408ad6` | Tracked |

This fixed cloud avoids variability in the scoring reference. It was derived
from the Replica office4 mesh, so its redistribution status depends on the
dataset terms and is **TBD**. If it cannot be shipped, the submission must
provide a deterministic generation procedure and record the generated cloud's
checksum.

Note that `stitch_office4.py` still independently samples the mesh to construct
per-window target pools. Keeping the fixed scoring cloud alone does not make the
entire evaluation deterministic.

## Source provenance and licenses

Reusable model and inference functionality lives in the project-level `vc3r`
package. The `vc3r_eval` package contains office4 benchmark orchestration,
metrics, validation, and result reporting. Neither package imports the
historical training, cache, or demo drivers.

| Component | Local source | License/provenance state |
|---|---|---|
| Project code | repository commit `34f002806ad91e8a62fa55c51012eb63cd5407fe` on `prep-main` | Project-level submission license is **TBD** |
| Depth Anything 3 | `da3/` | Apache-2.0; official submodule pinned to `41736238f5bced4debf3f2a12375d2466874866d` |
| SALAD | `da3/da3_streaming/loop_utils/salad/` | Nested DA3 submodule pinned to `6aede13a3f6c25750bf7fde10209c06cb73060bb` |
| NOVA3R | `nova3r/` | Apache-2.0 for NOVA3R-authored code; official submodule pinned to `b2818ea4928f169761573c8b3182730405de174d` |
| TripoSG subset | `nova3r/third_party/triposg/` | MIT; retained notice at `docs/submission/licenses/TRIPOSG_LICENSE.txt` |
| CroCo subset | `nova3r/croco/` | Contains upstream license/notice files; retain those with any shipped source |
| DUST3R subset | `nova3r/dust3r/` | CC BY-NC-SA 4.0; retained notice at `docs/submission/licenses/DUST3R_LICENSE.txt` |
| ChamferDist custom | Not retained | The malformed and unused gitlink was removed during NOVA3R submodule conversion |

The DA3 and NOVA3R license files currently hash to:

```text
c78446e29c48900cda82620a8df183cca61f0a595e05a49d0401a5fd604dd1870  da3/LICENSE
27237297a227fc220964e70108a86f3aa98fcdbde8c61c9d92274feb270999f1  nova3r/LICENSE
```

The pin evidence is documented in `docs/submission/SUBMODULE_AUDIT.md`. Both
dependencies are structurally pinned but remain runtime-unverified until
checkpoint loading and regression validation run in the final environment.
Project-authored alignment and Replica functionality formerly located inside
DA3 now lives in `vc3r/`; no compatibility files remain inside the submodule.

## Ignored and untracked dependencies

A clean clone currently lacks all of the following:

- project adapter checkpoints;
- optional point-flow checkpoint;
- NOVA3R scene-AE checkpoint and Hydra configuration;
- DA3 model weights;
- Replica data; and
- ignored historical logs and most evaluation output directories.

The fixed office4 GT PLY is tracked, but its redistribution status remains
unresolved. Before cleanup, every required ignored artifact must either gain an
approved distribution mechanism or a documented external acquisition step.

## Pre-submission decisions

- Select the canonical adapter and whether point flow is included.
- Confirm public redistribution of project checkpoints.
- Add stable checkpoint download URLs if checkpoints remain external.
- Keep the NOVA3R checkpoint as an official upstream download unless its
  redistribution terms are clarified.
- Keep DA3-LARGE-1.1 as a pinned official download; its model repository labels
  it Apache-2.0.
- Do not distribute Replica or its derived GT cloud under the working policy;
  provide acquisition and deterministic generation instructions.
- Record exact upstream source revisions for vendored DA3 and NOVA3R code.
- Review repository-wide compatibility with retained CC BY-NC-SA 4.0 code.
- Resolve or remove the malformed ChamferDist gitlink.
- Add a project-level license if one is not already provided elsewhere.

See [`REDISTRIBUTION_POLICY.md`](REDISTRIBUTION_POLICY.md) for the working
release policy and authoritative upstream references.
