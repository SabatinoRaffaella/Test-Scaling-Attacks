from torch import nn


class DefenseWrapper(nn.Module):
        """
        Wrap a model to apply a preprocessing/defense transform before the forward pass.
        Example defenses:
          - simple input clipping / random resize / jitter
          - JPEG compression (if you implement or have a function)
          - channels-wise smoothing
        Provide a callable `defense_fn(inputs)` that returns modified inputs.
        """

        def __init__(self, base_model: nn.Module, defense_fn=None, name="noop"):
            super().__init__()
            self.base = base_model
            self.defense_fn = defense_fn or (lambda x, deterministic=False: x)
            self.name = name

        def forward(self, x):
            # If gradients are required, assume attack is running and use deterministic mode
            deterministic = getattr(x, "requires_grad", False)
            x_def = self.defense_fn(x, deterministic=deterministic)

            # Ensure differentiable
            if not x_def.requires_grad:
                x_def = x_def.clone().detach().requires_grad_(True)
               # print(f"[WARN] Defense '{self.name}' returned non-differentiable tensor. Re-enabled requires_grad.")

            return self.base(x_def)