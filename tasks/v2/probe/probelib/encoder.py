"""Frozen teacher encoder -> one CLS embedding per scan.

load_teacher() loads a run's teacher backbone (EMA weights, frozen); cls_token()
turns a 4D scan into a single 384-d vector = the mean CLS over sliding 270-frame
windows, reusing the SAME TR harmonization / z-score as training (imported from
dinov2.data.fmri_data so probe and training preprocess identically).
"""

from pathlib import Path

import torch
from omegaconf import OmegaConf

from dinov2.models import build_model_from_cfg
from dinov2.data.fmri_data import (_load_mmap, _temporal_resample,
                                   _zscore_per_frame, TARGET_TR)

T_FIXED = 270
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_teacher(run_dir, checkpoint):
    """Load the run's teacher backbone (EMA of the student), frozen and in eval mode."""
    cfg = OmegaConf.load(Path(run_dir) / "config.yaml")
    _student, teacher, _dim = build_model_from_cfg(cfg)
    ckpt = torch.load(Path(run_dir) / checkpoint, map_location="cpu", weights_only=False)
    state = ckpt.get("model", ckpt)
    tb = {k.replace("_fsdp_wrapped_module.", "")[len("teacher.backbone."):]: v
          for k, v in state.items()
          if k.replace("_fsdp_wrapped_module.", "").startswith("teacher.backbone.")}
    teacher.load_state_dict(tb, strict=False)
    return teacher.to(DEVICE).eval()


@torch.no_grad()
def cls_token(teacher, path, native_tr):
    """One 384-d embedding per scan = mean CLS over sliding native windows, each
    resampled to 270 frames @ 0.72 s (identical harmonization to training)."""
    win = max(1, round(T_FIXED * TARGET_TR / native_tr))
    stride = max(1, win // 2)
    scan = _load_mmap(path).float()
    if scan.ndim == 4:
        scan = scan.unsqueeze(1)                        # (T, 1, X, Y, Z)
    embs = []
    for s in range(0, max(scan.shape[0] - win + 1, 1), stride):
        clip = _zscore_per_frame(_temporal_resample(scan[s:s + win].clone(), T_FIXED))
        out = teacher(clip.unsqueeze(0).to(DEVICE), is_training=True)
        embs.append(out["x_norm_clstoken"].squeeze(0).float().cpu())
    return torch.stack(embs).mean(0).numpy()
