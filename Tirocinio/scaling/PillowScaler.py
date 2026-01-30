import numpy as np
import io

import torch
from PIL import Image
from torchvision import transforms
"""
Utility for resizing/compressing images with PIL while keeping them as torch tensors.
Provides consistent crop/pad back to original target size.
"""
class PillowScaler:
    def __init__(self, target_size):
        """
        Args:
            target_size: (H, W) tuple for the final output size
        """
        self.H, self.W = target_size

    def resize_tensor(self, tensor, scale: float):
        """
        Resize a [C, H, W] tensor with a scale factor and restore it to target size.
        """
        # convert to PIL, resize, then center-crop/pad back to (H,W)
        pil = transforms.ToPILImage()(tensor.cpu().clamp(0, 1))
        # compute new size (width, height) for PIL resize
        new_size = (
            max(1, int(self.W * scale)),
            max(1, int(self.H * scale)),
        )
        resized = pil.resize(new_size, resample=Image.BILINEAR)
        t = transforms.ToTensor()(resized)

        # Center crop or pad to restore original HxW
        if t.shape[1] >= self.H and t.shape[2] >= self.W:
            t = transforms.CenterCrop((self.H, self.W))(t)
        else:
            pad_h, pad_w = max(0, self.H - t.shape[1]), max(0, self.W - t.shape[2])
            t = transforms.Pad((0, 0, pad_w, pad_h))(t)

        return t

    @staticmethod
    def compress_pillow(img_tensor_chw: torch.Tensor, quality: int):
        """
        Compress a CHW uint8 image using Pillow’s real JPEG encoder.
        """
        # Ensure uint8
        if img_tensor_chw.dtype != torch.uint8:
            img_tensor_chw = (img_tensor_chw.clamp(0, 1) * 255).byte()

        # CHW → HWC
        img_hwc = img_tensor_chw.permute(1, 2, 0).cpu().numpy()

        # Convert to PIL Image
        pil_img = Image.fromarray(img_hwc)

        # JPEG compress into memory buffer
        buffer = io.BytesIO()
        pil_img.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)

        # Decode back to tensor
        pil_decoded = Image.open(buffer).convert("RGB")
        arr = np.array(pil_decoded).copy()  # HWC uint8
        decoded = torch.from_numpy(arr).permute(2, 0, 1)  # CHW

        return decoded