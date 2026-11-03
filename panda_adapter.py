"""Bridge-style physical delta actions to an explicitly world-frame Panda OSC.

This is an interface adapter, not a Panda-finetuned policy or calibration learned
from Panda demonstrations. Input: metres, extrinsic XYZ Euler radians, openness.
"""
import numpy as np
from scipy.spatial.transform import Rotation


class PandaAdapter:
    def __init__(self, controller):
        assert controller.input_type == "delta"
        assert controller.input_ref_frame == "world"
        self.input_min = np.asarray(controller.input_min, dtype=float)
        self.input_max = np.asarray(controller.input_max, dtype=float)
        self.output_min = np.asarray(controller.output_min, dtype=float)
        self.output_max = np.asarray(controller.output_max, dtype=float)
        self.scale = (self.output_max - self.output_min) / (self.input_max - self.input_min)
        self.input_center = (self.input_max + self.input_min) / 2
        self.output_center = (self.output_max + self.output_min) / 2
        assert self.scale.shape == (6,) and np.all(self.scale > 0)

    def convert(self, physical):
        physical = np.asarray(physical, dtype=float)
        assert physical.shape == (7,) and np.isfinite(physical).all()
        delta = np.r_[physical[:3], Rotation.from_euler("xyz", physical[3:6]).as_rotvec()]
        unbounded = (delta - self.output_center) / self.scale + self.input_center
        arm = np.clip(unbounded, self.input_min, self.input_max)
        # OXE: 1=open. Panda: -1=open, +1=close. Binary choice is explicit.
        gripper = -1.0 if physical[6] >= 0.5 else 1.0
        return np.r_[arm, gripper], int(np.count_nonzero(arm != unbounded))
