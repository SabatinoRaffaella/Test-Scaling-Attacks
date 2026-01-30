"""
training
========
Provides fine-tuning, training, evaluation, and model utilities for the
Tirocinio Image Scale Attacks project.
"""

__version__ = "0.1.0"
__author__ = "Your Name"

from .finetune import finetune_classifier
from .train_utils import setup_optimizer, setup_scheduler
from .eval_utils import evaluate
from .classifier_utilities import load_pretrained_model, freeze_backbone, unwrap_model, find_classifier_module, reset_classifier_to_n
from utils.path_handling import prepare_checkpoint_dir

__all__ = [
    "finetune_classifier",
    "load_pretrained_model",
    "setup_optimizer",
    "setup_scheduler",
    "freeze_backbone",
    "evaluate",
    "unwrap_model",
    "find_classifier_module",
    "reset_classifier_to_n",
    "prepare_checkpoint_dir"
]