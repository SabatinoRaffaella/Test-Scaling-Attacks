import torchvision.io as io

import torch
from torch import nn

from scaling.PillowScaler import PillowScaler
from utils.image_utilities import save_correct_adv_and_jpeg, save_correctly_classified_images

@torch.no_grad()
def evaluate(model, val_loader, device):
    model.eval()
    correct, total = 0, 0
    for x, y in val_loader:
        x, y = x.to(device), y.to(device)
        preds = model(x).argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)
    return correct / total if total > 0 else 0

# ---------------------------------------------------------
# EVALUATE LOADER
# ---------------------------------------------------------
def evaluate_loader(model: nn.Module, loader, device: torch.device, attack=None,
                    max_batches=None, return_adv_batches=False):
    """
    Evaluate clean + adversarial accuracy on loader.
    Returns: summary, rows, optionally adv batches.
    """
    model.eval()
    total_samples = 0
    total_clean_correct = 0
    total_adv_correct = 0
    total_success_on_correct = 0
    rows = []

    adv_batches = [] if return_adv_batches else None
    lbl_batches = [] if return_adv_batches else None

    for batch_idx, batch in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        imgs, labels = batch[:2]
        imgs = imgs.to(device)
        labels = labels.to(device)
        batch_size = imgs.size(0)
        total_samples += batch_size

        # --- Clean predictions ---
        with torch.no_grad():
            logits = model(imgs)
            clean_preds = logits.argmax(dim=1)
            clean_correct = (clean_preds == labels).sum().item()
            total_clean_correct += clean_correct

        adv_preds = None
        adv_correct = 0
        successful_on_correct = 0

        if attack is not None:
            adv_imgs = attack(imgs, labels).to(device)
            with torch.no_grad():
                adv_logits = model(adv_imgs)
                adv_preds = adv_logits.argmax(dim=1)
                adv_correct = (adv_preds == labels).sum().item()
                total_adv_correct += adv_correct
                successful_on_correct = ((clean_preds == labels) & (adv_preds != labels)).sum().item()
                total_success_on_correct += successful_on_correct

            if return_adv_batches:
                adv_batches.append(adv_imgs.cpu())
                lbl_batches.append(labels.cpu())

            del adv_imgs, adv_logits, adv_preds
            torch.cuda.empty_cache()

        rows.append({
            "batch_idx": batch_idx,
            "batch_size": batch_size,
            "clean_correct": clean_correct,
            "adv_correct": adv_correct,
            "successful_on_correct": successful_on_correct
        })

# ---------------------------------------------------------
# JPEG RECOVERY FUNCTION (post-attack only)
# ---------------------------------------------------------
@torch.no_grad()
def evaluate_jpeg_recovery(model: nn.Module, attacked_images: torch.Tensor,
                           labels: torch.Tensor, compression_levels: list,
                           normalize_fn: callable,
                           clean_correct,
                           adv_correct,
                           save_dir,
                           prefix="",
                           save_adv_and_jpeg=False):
    """
    Evaluate model accuracy after compressing attacked images at multiple JPEG quality levels.

    Args:
        model: nn.Module, the model to evaluate
        attacked_images: [B, C, H, W] float tensor in [0,1]
        labels: [B] long tensor
        compression_levels: list of JPEG quality integers (10-100)

    Returns:
        dict: {quality_level: accuracy}
    """
    #print("DEBUG: function entered", flush=True)
    results = {}

    # Converte le immagini attaccate in uint8 per l'encoding JPEG
    # (PILLOW usa la CPU per la compressione di immagini)
    attacked_uint8 = (attacked_images.clamp(0, 1) * 255).round().byte().cpu()
    #print("DEBUG attacked_uint8:", attacked_uint8.shape, attacked_uint8.dtype,
    #      attacked_uint8.min(), attacked_uint8.max())

    for q in compression_levels:
        device = next(model.parameters()).device
        labels = labels.to(device)

        decoded_list = []
        for img in attacked_uint8:
            decoded = PillowScaler.compress_pillow(img_tensor_chw=img,quality=q)
            decoded_list.append(decoded.to(device))

        decoded_batch = torch.stack(decoded_list).float() / 255.

        # 5. NORMALIZE the data to match model's training parameters
        normalized_input = normalize_fn(decoded_batch)

        # --- JPEG predictions ---
        outputs = model(normalized_input)
        jpeg_preds = outputs.argmax(dim=1)
        jpeg_correct = jpeg_preds == labels

        # --- JPEG accuracy ---
        acc = jpeg_correct.float().mean().item()
        
        if save_adv_and_jpeg:
          recovered_mask = clean_correct & (~adv_correct) & jpeg_correct  

          save_correct_adv_and_jpeg(
              adv_images=attacked_images,
              jpeg_images=decoded_batch,
              correct_mask=recovered_mask,
              save_dir=save_dir,
              prefix=prefix,
              quality=q,
              max_samples=1,
          )
        results[q] = acc
        #print(f"DEBUG quality {q}: accuracy {acc}")

    return results