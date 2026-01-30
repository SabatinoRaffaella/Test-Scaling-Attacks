import torch
import torch.nn as nn
class NormalizedModel(nn.Module):
    def __init__(self, base_model, mean, std):
        super().__init__()
        self.register_buffer("_mean", torch.tensor(mean).view(1,3,1,1))
        self.register_buffer("_std", torch.tensor(std).view(1,3,1,1))
        self.base = base_model

    def forward(self, x):
        # normalize inside the model
        x = (x - self._mean) / self._std
        return self.base(x)