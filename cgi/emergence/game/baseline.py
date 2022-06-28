import torch


class SimpleBaseline:
    baseline: float
    count: int

    def __init__(self):
        self.baseline = 0.0
        self.count = 0

    def __call__(self):
        return self.baseline

    def update(self, x: torch.Tensor):
        self.count += 1
        self.baseline += (
            x.detach().mean().item() - self.baseline
        ) / self.count
