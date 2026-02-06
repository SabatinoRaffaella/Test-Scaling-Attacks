import torch
from torch import nn
from tqdm import tqdm
from utils.path_handling import prepare_checkpoint_dir
from .classifier_utilities import (
    freeze_backbone,
    reset_classifier_to_n,
    unwrap_model, find_classifier_module
)

from .train_utils import setup_optimizer, setup_scheduler
from .eval_utils import evaluate
from typing import Optional

def finetune_classifier(model: nn.Module,
                        train_loader,
                        val_loader,
                        device,
                        epochs: int = 5,
                        lr: float = 1e-3,
                        freeze_backbone_flag: bool = True,
                        n_classes: Optional[int] = None,
                        model_name: str = "model",
                        out_dir: str = None):
    """
    Fine-tune the classifier head. This function unwraps wrappers, ensures the classifier
    exists with correct number of classes, freezes backbone if requested, and trains the head.
    - model: possibly wrapped (NormalizedModel, DefenseWrapper etc.)
    - train_loader/val_loader must match required transforms (ToTensor() in [0,1])
    - n_classes: optional override; if not given, attempt to infer from classifier
    """
    # --- Ensure proper device ---
    device = torch.device(device) if isinstance(device, str) else device

    # 🔹 Use shared path utility
    out_dir, ckpt_path = prepare_checkpoint_dir(out_dir, model_name)

    # --- Load checkpoint if exists ---
    if ckpt_path.exists():
        print(f"[INFO] Loading checkpoint from {ckpt_path}")
        state = torch.load(ckpt_path, map_location=device)
        unwrap_model(model).load_state_dict(state)
        model.to(device)
        return model

    # --- Training logic below ---
    print (f"[INFO] No checkpoint found for '{model_name}', training for {epochs} epochs...")
    # --- Unwrap to base model ---
    base = unwrap_model(model)

        # --- Infer n_classes if not provided ---
    if n_classes is None:
        if hasattr(train_loader.dataset, "classes"):
            n_classes = len(train_loader.dataset.classes)
            print(f"[INFO] Inferred n_classes={n_classes} from dataset")
        else:
            raise RuntimeError("Cannot infer number of classes from dataset")

    # --- Reset classifier ---
    reset_classifier_to_n(base, n_classes)
    base.num_classes = n_classes
    base.to(device)  

    # --- Optimizer setup ---
    if freeze_backbone_flag:
        freeze_backbone(model)

    optimizer = setup_optimizer(model, lr)
    scheduler = setup_scheduler(optimizer)
    loss_fn = torch.nn.CrossEntropyLoss()

    print("[DEBUG] Starting training loop")
    # --- Training loop ---
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
          x, y = x.to(device), y.to(device)
          optimizer.zero_grad()
          loss = loss_fn(model(x), y)
          loss.backward()
          optimizer.step()
          total_loss += loss.item()

        scheduler.step()
        val_acc = evaluate(model, val_loader, device)
        print(f"[Epoch {epoch:03d}] loss={total_loss / len(train_loader):.4f} val_acc={val_acc:.4f}")


    # --- Save checkpoint ---
    torch.save(base.state_dict(), ckpt_path)
    print(f"[INFO] Saved checkpoint for {model_name} to {ckpt_path}")

    return model