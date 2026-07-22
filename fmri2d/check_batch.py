"""Show exactly how a SubjectSliceFolder batch is built: for a few subjects, print the slice
positions (z) available in the pool and the ones the transform samples as the 2 global crops
(what the teacher sees) + the local crops. Proves the batch really contains varied positions.

Run (in torch_env):
  DINO_SLICE_WINDOW=6 python3 fmri2d/check_batch.py /sci/labs/arieljaffe/dan.abergel1/brain2d_pool
"""
import os
import random
import sys

sys.path.insert(0, os.getcwd())
from dinov2.data.datasets import SubjectSliceFolder
from dinov2.data.augmentations import _slice_z

root = sys.argv[1]
n_show = int(sys.argv[2]) if len(sys.argv) > 2 else 12
need = 2 + int(os.environ.get("LOCAL_CROPS", "8"))     # 2 global + local crops
window = int(os.environ.get("DINO_SLICE_WINDOW", "0"))

ds = SubjectSliceFolder(root=root)
print(f"window={window}  crops/subject={need}  (0 window = fully random)\n")

globals_z = []
for i in range(min(n_show, len(ds.subjects))):
    sid, paths = ds.subjects[i]
    ordered = sorted(paths, key=_slice_z)
    if window > 0 and len(ordered) >= need:
        w = max(need, min(window, len(ordered)))
        start = random.randint(0, len(ordered) - w)
        chosen = random.sample(ordered[start:start + w], need)
    else:
        chosen = random.sample(ordered, need)
    cz = sorted(_slice_z(p) for p in chosen)
    g = cz[:2]                                          # the 2 global crops the teacher sees
    globals_z += g
    print(f"subj {sid}: pool z={sorted(set(_slice_z(p) for p in paths))}")
    print(f"            sampled z={cz}   ->  GLOBAL crops z={g}")

print(f"\nGLOBAL-crop positions across these {n_show} subjects (= what the teacher/centering sees):")
print(f"  {sorted(globals_z)}")
print(f"  distinct positions: {len(set(globals_z))} / {len(globals_z)}  "
      f"(spread {min(globals_z)}..{max(globals_z)})")
print("  -> if these are spread out, the batch is varied and the target should NOT be uniform.")
