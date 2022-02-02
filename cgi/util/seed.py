import torch
import random
import numpy.random as nprandom


def set_random_seed(seed: int) -> None:
    torch.manual_seed(seed)
    random.seed(seed)
    nprandom.seed(seed)
