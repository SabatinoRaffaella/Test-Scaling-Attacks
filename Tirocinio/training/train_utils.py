# training/train_utils.py
import torch.optim as optim

def setup_optimizer(model, lr=1e-3, weight_decay=0.0):
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(params, lr=lr, weight_decay=weight_decay)
    return optimizer


def setup_scheduler(optimizer, step_size=5, gamma=0.5):
    return optim.lr_scheduler.StepLR(optimizer, step_size, gamma)