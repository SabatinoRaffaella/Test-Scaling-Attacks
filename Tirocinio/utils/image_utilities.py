import numpy as np
import os
from PIL import Image
import torch
import matplotlib.pyplot as plt
from torchvision import transforms, utils as tv_utils
import torch.nn.functional as F
import pandas as pd

import random

def open_image(image_path):
    image = Image.open(image_path)
    # Show Image
    image.show()
    # print type of image
    print(type(image))
    # check property of image
    print(image.format, image.size, image.mode)
    # convert numpy array
    image_np = np.asarray(image)
    print(type(image_np))


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)
def denormalize_tensor(t: torch.Tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD):
    """
    Input: t tensor of shape [B,C,H,W] or [C,H,W], normalized with mean/std.
    Output: tensor same shape but denormalized to [0,1], clipped.
    """
    single = False
    if t.dim() == 3:
        t = t.unsqueeze(0)
        single = True
    device = t.device
    mean_t = torch.tensor(mean, device=device).view(1, -1, 1, 1)
    std_t  = torch.tensor(std,  device=device).view(1, -1, 1, 1)
    t = t * std_t + mean_t
    t = t.clamp(0.0, 1.0)
    return t.squeeze(0) if single else t

def normalize_tensor(x: torch.Tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD):
    """
    x: tensor in [0,1], returns normalized tensor for model input.
    Works whether mean/std are tuples or tensors.
    """
    single = False
    if x.dim() == 3:
        x = x.unsqueeze(0)
        single = True
    device = x.device
    mean_t = torch.tensor(mean, device=device).view(1, -1, 1, 1)
    std_t  = torch.tensor(std,  device=device).view(1, -1, 1, 1)
    out = (x - mean_t) / std_t
    return out.squeeze(0) if single else out

def tensor_to_pil(t: torch.Tensor):
    """Convert a [C,H,W] tensor in [0,1] to PIL Image."""
    t_cpu = t.detach().cpu()
    # torchvision.transforms.functional.to_pil_image expects [C,H,W] in [0,1]
    return transforms.functional.to_pil_image(t_cpu)

def ensure_dir(path: str):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def save_adv_examples(
    adv: torch.Tensor,
    orig: torch.Tensor,
    project_root: str,
    prefix: str = "attack",
    out_dirname: str = "outputs",
    mean=IMAGENET_MEAN,
    std=IMAGENET_STD,
    show_adv: bool = True,
):
    """
    Save adv/original/diff images to <project_root>/<out_dirname>/ and optionally open adv.

    adv, orig: tensors with shape [B,C,H,W] or [C,H,W]; typically B==1.
    project_root: path to project root (same value used in config.PROJECT_ROOT)
    prefix: filename prefix (e.g., 'coffee_run1')
    returns: dict with saved file paths
    """
    # ensure shapes are [B,C,H,W]
    if adv.dim() == 3:
        adv = adv.unsqueeze(0)
    if orig.dim() == 3:
        orig = orig.unsqueeze(0)

    # denormalize (assumes adv and orig are normalized for the model)
    orig_dn = denormalize_tensor(orig, mean=mean, std=std)

    # take first image in batch
    adv_img = adv
    orig_img = orig_dn[0]

    # compute absolute diff and amplify for visibility
    diff = (adv_img - orig_img).abs()
    # amplify small perturbations so they are visible; clip to [0,1]
    amp = 8.0  # you can tune this factor
    diff_vis = (diff * amp).clamp(0.0, 1.0)

    out_dir = os.path.join(project_root, out_dirname)
    ensure_dir(out_dir)

    orig_path = os.path.join(out_dir, f"orig_{prefix}.png")
    adv_path  = os.path.join(out_dir, f"adv_{prefix}.png")
    diff_path = os.path.join(out_dir, f"diff_{prefix}.png")

    # Save using torchvision utils (keeps range [0,1])
    tv_utils.save_image(orig_img, orig_path)
    tv_utils.save_image(adv_img, adv_path)
    tv_utils.save_image(diff_vis, diff_path)

    # Optionally open adv image with default viewer (desktop)
    if show_adv:
        try:
            pil = tensor_to_pil(adv_img)
            pil.show(title=f"Adversarial: {prefix}")
        except Exception:
            # show() can fail in headless environments; fail silently
            pass

    return {"orig": orig_path, "adv": adv_path, "diff": diff_path}


def save_correct_adv_and_jpeg(
        adv_images: torch.Tensor,  # [B,C,H,W] in [0,1]
        jpeg_images: torch.Tensor,  # [B,C,H,W] in [0,1]
        correct_mask: torch.Tensor,  # [B] bool
        save_dir: str,
        prefix: str = "",
        quality: int = None,
        max_samples: int = 5,
):
    """
      Save a few correctly classified images to disk.

      Args:
          images: [B, C, H, W] tensor in [0,1]
          correct_mask: boolean tensor [B] indicating correct predictions
          save_dir: base directory to save images
          prefix: optional prefix for filenames
          quality: optional quality level to organize subfolders
          max_samples: max images to save per batch/quality
      """
    os.makedirs(save_dir, exist_ok=True)

    if quality is not None:
        save_dir = os.path.join(save_dir, f"q{quality}")
        os.makedirs(save_dir, exist_ok=True)

    correct_indices = torch.nonzero(correct_mask).flatten().tolist()

    if len(correct_indices) == 0:
        return

    if len(correct_indices) > max_samples:
        correct_indices = random.sample(correct_indices, max_samples)

    for k, idx in enumerate(correct_indices):
        adv = adv_images[idx]
        jpeg = jpeg_images[idx]

        base = f"{prefix}_sample{k:03d}"

        tv_utils.save_image(
            adv, os.path.join(save_dir, f"{base}_adv.png")
        )
        tv_utils.save_image(
            jpeg, os.path.join(save_dir, f"{base}_jpeg.png")
        )


def save_correctly_classified_images(images: torch.Tensor,
                                     correct_mask: torch.Tensor,
                                     save_dir: str,
                                     prefix: str = "",
                                     quality: int = None,
                                     max_samples: int = 5):
    """
    Save a few correctly classified images to disk.
    
    Args:
        images: [B, C, H, W] tensor in [0,1]
        correct_mask: boolean tensor [B] indicating correct predictions
        save_dir: base directory to save images
        prefix: optional prefix for filenames
        quality: optional quality level to organize subfolders
        max_samples: max images to save per batch/quality
    """
    os.makedirs(save_dir, exist_ok=True)
    # Optional subfolder for quality
    if quality is not None:
        save_dir = os.path.join(save_dir, f"q{quality}")
        os.makedirs(save_dir, exist_ok=True)

    # Get indices of correctly classified images
    correct_indices = torch.nonzero(correct_mask).flatten().tolist()
    # Sample a subset to avoid too many images
    if len(correct_indices) > max_samples:
        correct_indices = random.sample(correct_indices, max_samples)

    for i, idx in enumerate(correct_indices):
        img = images[idx]
        filename = f"{prefix}_img{idx}.png"
        path = os.path.join(save_dir, filename)
        # Save tensor as image (keep in [0,1])
        tv_utils.save_image(img, path)

def random_scale_batch(x, scale_low, scale_high,  deterministic=False):
    """
    Randomly scale each image in the batch between scale_low and scale_high.
    Args:
        x: Tensor [B, C, H, W] in range [0,1]
    Returns:
        Tensor [B, C, H, W] with random scale + crop/pad back to original size.
    """
    B, C, H, W = x.shape

    if deterministic:
        factors = torch.ones(B, device=x.device)
    else:
        factors = torch.rand(B, device=x.device) * (scale_high - scale_low) + scale_low

    out = []
    for i in range(B):
        new_H = max(1, int(H * factors[i].item()))
        new_W = max(1, int(W * factors[i].item()))
        img_scaled = F.interpolate(x[i:i+1], size=(new_H, new_W), mode='bilinear', align_corners=False)

        # center crop/pad back to H,W
        pad_H = max(H - new_H, 0)
        pad_W = max(W - new_W, 0)
        pad_top, pad_bottom = pad_H // 2, pad_H - pad_H // 2
        pad_left, pad_right = pad_W // 2, pad_W - pad_W // 2
        img_scaled = F.pad(img_scaled, (pad_left, pad_right, pad_top, pad_bottom))

        if new_H > H:
            start_H = (new_H - H) // 2
            img_scaled = img_scaled[:, :, start_H:start_H+H, :]
        if new_W > W:
            start_W = (new_W - W) // 2
            img_scaled = img_scaled[:, :, :, start_W:start_W+W]

        out.append(img_scaled)

    return torch.cat(out, dim=0)


def show_batch_imgrid(loader, mean, std, n=8):
    """Plotta le prime n immagini di un Dataloader """
    imgs, labels = next(iter(loader))
    imgs = imgs[:n]
    # unnormalize
    imgs = imgs * std.view(1,3,1,1) + mean.view(1,3,1,1)
    imgs = imgs.clamp(0,1).cpu().numpy()
    imgs = np.transpose(imgs, (0,2,3,1))  # B,H,W,C
    fig, axes = plt.subplots(1, n, figsize=(n*2,2))
    for i, ax in enumerate(axes):
        ax.imshow(imgs[i])
        ax.axis('off')
    plt.show()


def plot_eps_sweep(results_root, attack_name="FGSM", defense_name="none", model_name="efficientnet_b0", save_path=None):
    """
    Aggregate CSVs across epsilon values and plot clean / adversarial accuracy vs ε (in 1/255 scale).

    Args:
        results_root (str): Path to the main results folder, e.g. 'attack_results/'
        attack_name (str): Attack identifier ("FGSM", "PGD", etc.)
        defense_name (str): Defense name ("none", "gaussian", etc.)
        model_name (str): Model identifier ("efficientnet_b0", etc.)
        save_path (str, optional): If given, save the plot instead of showing it.
        Folder structure expected:
    attack_results/
        model__def-defense__atk-attack/
            eps-0.031/
                run_YYYYMMDD-HHMMSS/results.csv
    """
    folder_pattern = f"{model_name}__def-{defense_name}__atk-{attack_name}"
    eps_values, clean_acc, adv_acc = [], [], []

    for attack_folder in sorted(os.listdir(results_root)):
        if folder_pattern not in attack_folder:
            continue
        attack_path = os.path.join(results_root, attack_folder)

        # Loop over eps folders
        for eps_folder in sorted(os.listdir(attack_path)):
            if not eps_folder.startswith("eps-"):
                continue
            eps_val = float(eps_folder.replace("eps-", ""))

            eps_path = os.path.join(attack_path, eps_folder)

            # Find latest run
            run_dirs = [d for d in os.listdir(eps_path) if d.startswith("run_")]
            if not run_dirs:
                continue
            latest_run = sorted(run_dirs)[-1]
            csv_file = os.path.join(eps_path, latest_run, "results.csv")
            if not os.path.exists(csv_file):
                continue

            df = pd.read_csv(csv_file)
            total = df["batch_size"].sum()
            clean_correct = df["clean_correct"].sum()
            adv_correct = df["adv_correct"].sum()

            eps_values.append(eps_val)
            clean_acc.append(clean_correct / total)
            adv_acc.append(adv_correct / total)

    if not eps_values:
        print("[WARN] No results found for the given attack/defense/model combination.")
        return

    # Sort by epsilon
    eps_values, clean_acc, adv_acc = zip(*sorted(zip(eps_values, clean_acc, adv_acc)))

    # Plot
    plt.figure(figsize=(7, 5))
    plt.plot([e*255 for e in eps_values], [a*100 for a in clean_acc], "o--", label="Clean acc (%)")
    plt.plot([e*255 for e in eps_values], [a*100 for a in adv_acc], "o-", label="Adversarial acc (%)")
    plt.xlabel("ε (L∞, in /255)")
    plt.ylabel("Accuracy (%)")
    plt.title(f"{model_name} — {attack_name} vs ε (def={defense_name})")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=200)
        print(f"[INFO] Saved plot to {save_path}")
    else:
        plt.show()