"""Linear probe on a trained brain-DINOv2 checkpoint, using the OFFICIAL DINOv2 representation
(CLS of last blocks + avgpool patches, swept over n_blocks in {1,4} x avgpool, best config).
Same code path as the periodic probe (fmri2d/online_probe.probe_backbone), subject-level GroupKFold.

Labels come from a CSV: subject id parsed from PNG name (subject_<ID>_...png) -> --label-col
(default Gender, i.e. sex, chance 50%). Load a specific run's checkpoint via --output-dir.

Run (via probe_brain.sh):
  python probe_brain.py --config-file <cfg> --output-dir <run_dir> \
      --features-root <brain2d> --labels-csv <HCP_YA_subjects.csv> --label-col Gender --cv 5 \
      dino.head_n_prototypes=... ibot.head_n_prototypes=...
"""
import argparse

import torch

_orig_torch_load = torch.load
def _torch_load(*a, **k):
    k["weights_only"] = False
    return _orig_torch_load(*a, **k)
torch.load = _torch_load

from dinov2.train.train import get_args_parser
from dinov2.utils.config import setup
from dinov2.train.ssl_meta_arch import SSLMetaArch
from dinov2.fsdp import FSDPCheckpointer
from dinov2.models import build_model_from_cfg

from fmri2d.online_probe import probe_backbone


def main():
    ap = get_args_parser(add_help=True)
    ap.add_argument("--features-root", required=True, help="ImageFolder root with the brain PNGs")
    ap.add_argument("--labels-csv", required=True, help="HCP_YA_subjects.csv")
    ap.add_argument("--label-col", default="Gender", help="CSV column to predict (Gender=sex)")
    ap.add_argument("--id-col", default="Subject")
    ap.add_argument("--cv", type=int, default=5, help="subject-level GroupKFold folds")
    ap.add_argument("--test-frac", type=float, default=0.2, help="(deprecated, ignored)")
    ap.add_argument("--avgpool", action="store_true", help="(deprecated: representation is swept)")
    args = ap.parse_args()

    cfg = setup(args)
    model = SSLMetaArch(cfg).to(torch.device("cuda"))
    model.prepare_for_distributed_training()
    FSDPCheckpointer(model, cfg.train.output_dir).resume_or_load(cfg.MODEL.WEIGHTS, resume=True)

    # get_intermediate_layers doesn't work on the FSDP-wrapped backbone (len(self.blocks[-1]) fails).
    # Build a plain (non-FSDP) backbone and load the teacher's weights into it.
    backbone, _ = build_model_from_cfg(cfg, only_teacher=True)
    backbone = backbone.cuda().eval()
    sd = {k[len("backbone."):]: v for k, v in model.teacher.state_dict().items() if k.startswith("backbone.")}
    backbone.load_state_dict(sd, strict=True)

    print(f"probing {cfg.train.output_dir} on {args.features_root} (official repr, {args.label_col})...", flush=True)
    mean, std = probe_backbone(
        backbone, args.features_root, args.labels_csv,
        label_col=args.label_col, cv=(args.cv or 5), id_col=args.id_col,
    )
    print(f"\n============ BRAIN PROBE: {args.label_col} (official DINOv2 repr) ============")
    print(f"  CV accuracy: {mean * 100:.2f}% +/- {std * 100:.2f}%   ({args.cv}-fold, subject-level)")
    print("=================================================================")


if __name__ == "__main__":
    main()
