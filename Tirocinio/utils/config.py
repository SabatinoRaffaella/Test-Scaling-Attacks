# config.py
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Callable
import torch
import os

from utils.image_utilities import random_scale_batch

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))

@dataclass
class ModelConfig:
    model_name: str = "efficientnet_b0"
    pretrained: bool = True
    num_classes: int = 200

@dataclass
class AttackConfig:
    name: str = "AutoAttack"  # "FGSM", "PGD", "none" etc.
    params: Dict = field(default_factory=lambda: {
        "eps": 8 / 255.0,
        "alpha": 2 / 255.0,
        "steps": 40,
        "num_eot_samples": 8,
        "targeted": False
    })
    eps_list: Optional[List[float]] = field(default_factory=lambda: [2 / 255, 4 / 255, 8 / 255, 12 / 255, 16 / 255])

@dataclass
class DefenseConfig:
    name: str = "none"
    params: Dict = field(default_factory=dict)
    # Optional callable that takes a tensor [B,C,H,W] -> [B,C,H,W]
    fn: Optional[Callable] = None

@dataclass
class DataConfig:
    data_root: str = os.path.join(PROJECT_ROOT, "data", "tiny-imagenet-200")
    batch_size: int = 32
    # Model input size: optional; we can auto-detect from timm, but it's handy to keep here
    input_size: Optional[List[int]] = None  # [C,H,W] or None
    num_workers: int = 2

@dataclass
class RunConfig:
    # device detection is done at import/run time
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")
    seed: int = 1234
    out_dir: str = "attack_results"
    verbose: bool = True
    num_epochs: int = 10
    save_examples: bool = True
    max_batches: Optional[int] = None


@dataclass
class Config:
    run: RunConfig = field(default_factory=RunConfig)
    data: DataConfig = field(default_factory=DataConfig)

    # Lista dei Modelli
    models: List[ModelConfig] = field(default_factory=lambda: [
        ModelConfig("efficientnet_b0", pretrained=True, num_classes=200),
        ModelConfig("resnet18", pretrained=True, num_classes=200)
    ])

    #Lista di tipologie di attacchi scaling
    attacks: List[AttackConfig] = field(default_factory=lambda: [
        AttackConfig(name="FGSM", params={"eps": 8 / 255.0}),
        AttackConfig(name="PGD", params={"eps": 8 / 255.0, "alpha": 2 / 255.0, "steps": 20}),
        AttackConfig(name="AutoAttack", params={"eps": 8 / 255.0}),
    ])

    #Lista delle tecniche di difesa da attachi scaling
    defenses: List[DefenseConfig] = field(default_factory=lambda: [
        DefenseConfig(name="none", fn=lambda x, deterministic=False: x),
        DefenseConfig(name="Gaussiana", fn=lambda x: x + 0.01 * torch.randn_like(x)),
        DefenseConfig(name="Random_scale", fn=lambda x: random_scale_batch(x, 0.8, 1.2)),
    ])

# helper for convenience if you want a single global config object
cfg = Config()