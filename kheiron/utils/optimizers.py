import torch


def get_lr_strings(optimizer):
    return ",".join([f"{group['lr']:10.3e}" for group in optimizer.param_groups])


class OptimizerNames:
    AdamW = {'name': 'adamw', 'cls': torch.optim.AdamW}
    SGD = {'name': 'sgd', 'cls': torch.optim.SGD}


