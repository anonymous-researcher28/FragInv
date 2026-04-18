#didgen repo

import torch
from torch import Tensor
from torch.utils.data import Sampler
from typing import Sequence, Iterator

class SubsetWeightedRandomSampler(Sampler[int]):
    weights: Tensor
    num_samples: int
    replacement: bool

    def __init__(self, weights: Sequence[float],
                 indices: Sequence[int],
                 replacement: bool = True, generator=None) -> None:
        if not isinstance(replacement, bool):
            raise ValueError("replacement should be a boolean value, but got "
                             "replacement={}".format(replacement))

        weights_tensor = torch.as_tensor(weights, dtype=torch.double)
        if len(weights_tensor.shape) != 1:
            raise ValueError("weights should be a 1d sequence but given "
                             "weights have shape {}".format(tuple(weights_tensor.shape)))

        self.weights = weights_tensor[indices]
        self.num_samples = len(indices)
        self.indices = indices
        self.replacement = replacement
        self.generator = generator

    def __iter__(self) -> Iterator[int]:
        rand_tensor = torch.multinomial(self.weights, self.num_samples, self.replacement, generator=self.generator)
        yield from iter(self.indices[rand_tensor].tolist())

    def __len__(self) -> int:
        return self.num_samples
