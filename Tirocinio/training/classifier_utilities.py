import timm
from torch import nn
from typing import Tuple
from torchvision import models
# mapping from model_name → (constructor, weights_enum)
MODEL_REGISTRY = {
    "resnet18": (models.resnet18, models.ResNet18_Weights.IMAGENET1K_V1),
    "resnet50": (models.resnet50, models.ResNet50_Weights.IMAGENET1K_V2),
    "efficientnet_b0": (models.efficientnet_b0, models.EfficientNet_B0_Weights.IMAGENET1K_V1),
    "efficientnet_b1": (models.efficientnet_b1, models.EfficientNet_B1_Weights.IMAGENET1K_V2),

}

def load_pretrained_model(model_name: str, device="cuda"):
    """
    Load a torchvision model *with correct pretrained weights* and return:
       - model (eval & on device)
       - transforms matching the weights (resize/crop/normalize)
       - weights metadata (contains categories etc.)
    """
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Model '{model_name}' not found in registry!")

    constructor, weights_enum = MODEL_REGISTRY[model_name]

    # Load pretrained weights
    weights = weights_enum
    model = constructor(weights=weights)

    # Move to device and eval mode
    model = model.to(device).eval()

    # Build transform pipeline matching the weights
    preprocess = weights.transforms()

    return model, preprocess, weights


def make_model(name: str, num_classes: int, pretrained=True, device=None):
    """
    Create a timm model with the correct number of classes.
    Uses timm's built-in reset_classifier to handle EfficientNet, ResNet, ViT, ecc.
    """
    # Crea il modello con timm
    model = timm.create_model(name, pretrained=pretrained)

    if device:
        model = model.to(device)
    return model


def get_classifier_module(model):
    """
    Return (module, name) for the classifier/head of a timm model.
    Tries common timm API patterns.
    Proviamo a recuperare la testa del classificatore per bloccare
    ("freezare") l'apprendimento sulle componenti più profonde del
    classificatore che compongono i layer convoluzionali.
    """
    # timm has get_classifier() / reset_classifier() helpers on many models
    if hasattr(model, "get_classifier"):
        try:
            cls = model.get_classifier()
            # find parent module that contains this classifier object
            for name, module in model.named_children():
                if getattr(model, name) is cls:
                    return getattr(model, name), name
            # fallback: return the raw classifier object
            return cls, "classifier"
        except Exception:
            pass

    # common attribute names fallback
    # Cerca di recuperare la testa iterando su una lista
    # di nomi frequentemente usati.
    for name in ("head", "fc", "classifier", "head.fc"):
        parts = name.split(".")
        mod = model
        ok = True
        for p in parts:
            if hasattr(mod, p):
                mod = getattr(mod, p)
            else:
                ok = False
                break
        if ok:
            return mod, parts[0]  # return top-level attr name

    raise RuntimeError("Could not find classifier/head module automatically. Inspect the model to find its final layer.")


def unwrap_model(model: nn.Module) -> nn.Module:
    """Return the underlying base model if wrapped (NormalizedModel, DefenseWrapper, DataParallel, etc.)."""
    # common wrapper attrs used in your codebase
    for attr in ("base", "base_model", "module"):
        if hasattr(model, attr):
            inner = getattr(model, attr)
            # avoid infinite loop
            if inner is not model:
                return unwrap_model(inner)
    return model

def find_classifier_module(model: nn.Module) -> Tuple[nn.Module, str, nn.Module]:
    """
    Find the classifier module in a model.
    Returns (parent_module, attr_name_or_index, classifier_module).
    Raises RuntimeError if not found.
    """
    # 1) prefer timm helper if available
    try:
        cls = model.get_classifier()
        # find fullname of cls in named_modules
        modules = dict(model.named_modules())
        for fullname, module in modules.items():
            if module is cls:
                parent_name = fullname.rsplit('.', 1)[0]  # '' => root
                parent = modules.get(parent_name, model)
                child_name = fullname.rsplit('.', 1)[1]
                return parent, child_name, cls
    except Exception:
        pass

    # 2) scan for common attribute names
    for name in ("classifier", "head", "fc", "last_linear", "logits"):
        if hasattr(model, name):
            module = getattr(model, name)
            # if it's a sequential container, try last element if linear
            if isinstance(module, nn.Sequential) and len(module) > 0 and isinstance(module[-1], nn.Linear):
                return module, str(len(module)-1), module[-1]
            if isinstance(module, nn.Linear) or isinstance(module, nn.Sequential):
                return model, name, module

    # 3) fallback: take the last nn.Linear found in named_modules()
    modules = dict(model.named_modules())  # fullname -> module
    last_linear_fullname = None
    for fullname, module in modules.items():
        if isinstance(module, nn.Linear):
            last_linear_fullname = fullname

    if last_linear_fullname:
        modules_map = modules
        fullname = last_linear_fullname
        parent_name = fullname.rsplit('.', 1)[0]  # '' -> root
        parent = modules_map.get(parent_name, model)
        child_name = fullname.rsplit('.', 1)[1]
        cls = modules_map[fullname]
        return parent, child_name, cls

    # nothing found
    raise RuntimeError("Unable to find classifier/head in model (searched get_classifier(), common names, last Linear).")

def replace_module_in_parent(parent: nn.Module, child_name: str, new_module: nn.Module) -> None:
    """
    Replace parent's attribute or Sequential/index child named child_name with new_module.
    child_name may be numeric (index in Sequential) or attribute name.
    """
    # numeric index => Sequential, ModuleList, etc.
    if isinstance(parent, (nn.Sequential, nn.ModuleList)):
        try:
            idx = int(child_name)
            parent[int(idx)] = new_module
            return
        except Exception:
            pass

    # attribute replacement
    if hasattr(parent, child_name):
        setattr(parent, child_name, new_module)
        return

    # last resort: try mapping by name among children
    for name, module in parent.named_children():
        if name == child_name:
            setattr(parent, name, new_module)
            return

    raise RuntimeError(f"Unable to replace child '{child_name}' on parent {parent}.")

def reset_classifier_to_n(model: nn.Module, n_classes: int) -> None:
    """
    Use timm's reset_classifier if available, else replace detected classifier with a new Linear(n_in, n_classes).
    """
    # try timm reset_classifier
    try:
        model.reset_classifier(n_classes)
        return
    except Exception:
        pass

    parent, child_name, cls = find_classifier_module(model)
    # try to infer in_features
    in_features = None
    if isinstance(cls, nn.Linear) and hasattr(cls, "in_features"):
        in_features = cls.in_features
    elif hasattr(cls, "weight"):
        in_features = cls.weight.shape[1]
    else:
        raise RuntimeError("Cannot infer classifier in_features to create new head.")

    new_head = nn.Linear(in_features, n_classes)
    replace_module_in_parent(parent, child_name, new_head)


def freeze_backbone(model):
    for name, param in model.named_parameters():
        if "classifier" not in name and "fc" not in name:
            param.requires_grad = False
    return model