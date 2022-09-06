import random
import numpy as np
import torch


def set_ramdom_seed(seed: int):
    if seed > 0:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        # some cudnn methods can be random even after fixing the seed
        # unless you tell it to be deterministic
        torch.backends.cudnn.deterministic = True

    if torch.cuda.is_available() and seed > 0:
        torch.cuda.manual_seed_all(seed)
