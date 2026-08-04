"""Random-backbone baseline for the fMRI probe.

Builds a RANDOM (untrained) 4D ViT from the training config and runs the exact
periodic official linear probe (HCP Sex, full test set) on it — the mandatory
baseline for interpreting any trained probe number (see docs/paper_findings §1).

If random ~= trained -> SSL adds nothing / the label is saturated.
If trained << random -> SSL is degrading the representation (collapse).
If random ~= 50%      -> the probe pipeline itself is broken.

Run (via random_baseline.sh):
  python tasks/v2/probe/random_baseline.py --config-file dinov2/configs/train/fmri_vits.yaml \
      --output-dir /tmp/rand_probe
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))          # make probelib importable

from dinov2.train.train import get_args_parser
from dinov2.utils.config import setup
from dinov2.models import build_model_from_cfg
from dinov2.eval.fmri_periodic_probe import run_periodic_probe


def main():
    args = get_args_parser(add_help=True).parse_args()
    cfg = setup(args)
    # Random-init 4D backbone (teacher tower), same architecture/T as training.
    _student, teacher, _dim = build_model_from_cfg(cfg)
    teacher = teacher.cuda().eval()
    t_fixed = cfg.student.fmri_temporal_size
    print(f"RANDOM baseline: T_fixed={t_fixed}, arch={cfg.student.arch}, patch={cfg.student.patch_size}", flush=True)
    res = run_periodic_probe(teacher, torch.device("cuda"), t_fixed=t_fixed)
    print("\n============ RANDOM-BACKBONE BASELINE (HCP Sex, official probe) ============")
    if res:
        print(f"  test_acc={res['test_acc']:.4f}  val_acc={res['val_acc']:.4f}  "
              f"[best {res['best']}]  n_test={res['n_test']}")
    else:
        print("  probe returned None (HCP data unavailable)")
    print("============================================================================")


if __name__ == "__main__":
    main()
