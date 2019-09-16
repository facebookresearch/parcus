# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import random
from itertools import repeat
import numpy as np
import torch
from torch._six import container_abcs


# Let's create a default collate function to concatenate on the axis of words pairs and
# also return the batch object. Take code from default_collate
from torch.utils.data.dataset import Dataset
from torch.utils.data.sampler import WeightedRandomSampler


def custom_collate(batch):
    r"""Puts each data field into a tensor with outer dimension batch size"""

    error_msg_fmt = "batch must contain tensors, numbers, dicts or lists; found {}"
    elem_type = type(batch[0])

    if isinstance(batch[0], container_abcs.Sequence):
        # This will be called first
        transposed = zip(*batch)
        return [custom_collate(samples) for samples in transposed]
    elif isinstance(batch[0], torch.Tensor):
        # This will be called by the previous guard with tail recursion

        # If we're in a background process, concatenate directly into a
        # shared memory tensor to avoid an extra copy
        numel = sum([x.numel() for x in batch])
        storage = batch[0].storage()._new_shared(numel)
        out = batch[0].new(storage)

        # I HAVE TO ADDRESS BATCHING HERE
        # print(f'batch shapes are {[batch[i].shape for i in range(len(batch))]}')
        torch.cat(batch, dim=0, out=out)
        # print(f'final shape is {out.shape}')

        # NOW I NEED TO CREATE THE BATCH IDXES.
        batch_idxs = torch.zeros(batch[0].shape[0], dtype=torch.int).unsqueeze(dim=1)
        for i in range(1, len(batch)):
            batch_idxs = torch.cat((batch_idxs,
                                    torch.full((batch[i].shape[0], 1), i, dtype=torch.int)),
                                   dim=0)

            # I MUST ALSO PASS THE BATCH OBJECT
        return out, batch_idxs.squeeze()

    else:
        raise TypeError((error_msg_fmt.format(type(batch[0]))))


# Define auxiliary functions (code taken from torch_scatter, because I cannot load the package on Bento machine)
def maybe_dim_size(index, dim_size=None):
    if dim_size is not None:
        return dim_size
    return index.max().item() + 1 if index.numel() > 0 else 0


def gen(src, index, dim=-1, out=None, dim_size=None, fill_value=0):
    dim = range(src.dim())[dim]  # Get real dim value.

    # Automatically expand index tensor to the right dimensions.
    if index.dim() == 1:
        index_size = list(repeat(1, src.dim()))
        index_size[dim] = src.size(dim)
        index = index.view(index_size).expand_as(src)

    # Generate output tensor if not given.
    if out is None:
        out_size = list(src.size())
        dim_size = maybe_dim_size(index, dim_size)
        out_size[dim] = dim_size
        out = src.new_full(out_size, fill_value)

    return src, out, index, dim


def rusty_scatter_add(src, index, dim=-1, out=None, dim_size=None, fill_value=0):
    src, out, index, dim = gen(src, index, dim, out, dim_size, fill_value)
    return out.scatter_add_(dim, index, src)


class SingleRunWeightedRandomSampler(WeightedRandomSampler):
    """
    Instead of resampling at each epoch, we compute the samples indices one time only, and reuse them.
    We can shuffle in case the training uses minibatches of data
    """

    def __init__(self, weights, num_samples, replacement=True):
        super().__init__(weights, num_samples, replacement)
        self.indices = torch.multinomial(self.weights, self.num_samples, self.replacement).tolist()

    def __iter__(self):
        # Use numpy and not random, so that datasets are created in the same way
        np.random.shuffle(self.indices)  # operation is in-place
        return iter(self.indices)

    def __len__(self):
        return self.num_samples


class SubsetNoisy(Dataset):
    """
    Subset of a dataset at specified indices.

    Arguments:
        dataset (Dataset): The whole Dataset
        indices (sequence): Indices in the whole set selected for subset
    """
    def __init__(self, dataset, indices, noisy_targets, target_idx):
        self.dataset = dataset
        self.indices = indices
        self.target_idx = target_idx
        self.noisy_targets = noisy_targets

    def __getitem__(self, idx):
        embeddings, annotations, mask_input, target, indices = self.dataset[self.indices[idx]]
        return embeddings, annotations, mask_input, (torch.tensor([self.noisy_targets[idx]])+1)/2, indices

    def __len__(self):
        return len(self.indices)
