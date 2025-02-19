from typing import Optional
import warnings
import torch


class BaseModel:
    def __init__(self, precision: Optional[int] = 32):
        assert precision in [32, 64], 'Precision must be either 32 or 64'
        self.precision = torch.float32 if precision == 32 else torch.float64

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def to(self, device):
        warnings.warn('Method `to` is not implemented for this model. Returning the model as is.')
        return self
