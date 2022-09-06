import torch


class OptimizerNames:
    AdamW = {'name': 'adamw', 'cls': torch.optim.AdamW}
    SGD = {'name': 'sgd', 'cls': torch.optim.SGD}
