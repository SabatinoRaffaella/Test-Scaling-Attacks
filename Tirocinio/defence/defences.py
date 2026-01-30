import torch
import torch.nn.functional as F
from torch import nn
import io
from PIL import Image
from torchvision.transforms.functional import to_pil_image, to_tensor

def random_scale(x, min_factor=0.75, max_factor=1.25):
    """
    Randomly scales images in a differentiable way.
    x: tensor [B, C, H, W]
    """
    B, C, H, W = x.shape
    factor = torch.rand(B, 1, 1, 1, device=x.device) * (max_factor - min_factor) + min_factor
    return x * factor  # element-wise multiply, differentiable


def gaussian_noise(x, std=0.05):
    """
    Adds Gaussian noise in a differentiable way.
    """
    noise = torch.randn_like(x) * std
    return x + noise


def jpeg_compression(x, quality=75):
    """
    Approximate differentiable JPEG compression using blur + rounding approximation.
    Note: True JPEG with Python libraries is non-differentiable.
    """
    # Example: apply slight smoothing as proxy for JPEG artifacts
    kernel_size = 3
    x_blur = F.avg_pool2d(x, kernel_size, stride=1, padding=1)
    return x_blur

        

def clip_inputs(x, min_val=0.0, max_val=1.0):
    """Clips inputs in-place safely."""
    return torch.clamp(x, min_val, max_val)

def make_jpeg_defense(quality):
    #jpeg_def = JPEGCompressionDefense(quality=quality)

    def defense_fn(x, deterministic=False):
        return defense_fn
#return jpeg_def(x)

def build_defense_fn(def_cfg):
    name = def_cfg.name

    if name == "none":
        return lambda x, deterministic=False: x

    if name == "Gaussiana":
        return lambda x, deterministic=False: gaussian_noise(x)

    if name == "Random_scale":
        return lambda x, deterministic=False: random_scale(x)

    if name == "Compressione_JPEG":
        quality = def_cfg.params.get("quality", 75)
        return make_jpeg_defense(quality)

    raise ValueError(f"Unknown defense: {name}")    