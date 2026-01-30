import torch
import torchattacks
from torch import nn


def attack_factory(name: str, model: nn.Module, device: torch.device, eps=8/255, steps=40, alpha=None, **kwargs):
    """
    Return an attack object/function that can be called as adv = atk(images, labels).
    Supports some common torchattacks and AutoAttack 'standard' wrapper (if installed).
    `name` examples: "AutoAttack", "PGD", "FGSM", "CW", "BIM", "DeepFool" (depends on torchattacks package).
    kwargs: extra hyperparams per-attack.
    """
    name = name.lower()
    if name == "autoattack":
        # torchattacks.AutoAttack signature your project used:
        atk = torchattacks.AutoAttack(
            model,
            'Linf',
            eps,
            'standard',
            kwargs.get('num_classes', getattr(model, 'num_classes', None) or kwargs.get('num_classes', 1000)),
            kwargs.get('seed', 0),
            kwargs.get('verbose', False)
        )
        # some versions support set_device:
        try:
            atk.set_device(device)
        except Exception:
            pass
        return atk

    if name in ("FGSM","fgsm"):
        return torchattacks.FGSM(model, eps)
    if name in ("PGD","pgd", "pgd_linf", "pgd-linf"):
        alpha = alpha or (eps / steps if steps > 0 else eps / 4)
        return torchattacks.PGD(model, eps, alpha, steps)
    if name == "bim":
        alpha = alpha or (eps / steps if steps > 0 else eps / 4)
        return torchattacks.BIM(model, eps, alpha, steps)
    if name == "cw":
        # Carlini-Wagner commonly L2, needs extra args
        return torchattacks.CW(model, c=kwargs.get('c', 1e-4), kappa=kwargs.get('kappa', 0), steps=kwargs.get('steps', 1000))
    if name == "deepfool":
        return torchattacks.DeepFool(model, steps=kwargs.get('steps', 50))
    # Fallback: try to get from torchattacks by name
    try:
        AttClass = getattr(torchattacks, name.upper())
        return AttClass(model, **{k: v for k, v in kwargs.items()})
    except Exception:
        raise ValueError(f"Unknown attack: {name}")