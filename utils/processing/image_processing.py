import numpy as np
import torch

# Preprocesses an observation inplace (from float32 Tensor [0, 255] to [-0.5, 0.5])
@torch.jit.script
def normalize_image(observation, bit_depth):
  # Quantise to given bit depth and centre
  observation.div_(2 ** (8 - bit_depth)).floor_().div_(2 ** bit_depth).sub_(0.5)
  # Dequantise (to approx. match likelihood of PDF of continuous images vs. PMF of discrete images)
  observation.add_(torch.rand_like(observation).div_(2 ** bit_depth))
  return observation


# Postprocess an observation for storage (from float32 numpy array [-0.5, 0.5] to uint8 numpy array [0, 255])
def reverse_normalized_image(observation, bit_depth=5):
  return np.clip(np.floor((observation + 0.5) * 2 ** bit_depth) * 2 ** (8 - bit_depth), 0, 2 ** 8 - 1).astype(np.uint8)

