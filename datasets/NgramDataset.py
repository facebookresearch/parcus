# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import os
import torch
from pathlib import Path
from torch.utils.data import Dataset


class NGramsDataset(Dataset):

    def __init__(self, samples_folder, cache_data=True):
        self.folder = samples_folder
        self.cache_data = cache_data
        self.files = [filename for filename in sorted(os.listdir(samples_folder)) if '.torch' in filename]
        self.cached = [None for _ in range(len(self.files))]

    def __len__(self):
        # no of examples
        return len(self.files)

    def __getitem__(self, idx):

        if not self.cache_data:
            example, target = torch.load(Path(self.folder, self.files[idx]))
        else:
            if self.cached[idx] is None:
                example, target = torch.load(Path(self.folder, self.files[idx]))
                self.cached[idx] = (example, target)
            else:
                example, target = self.cached[idx]

        target = (target + 1) / 2

        # print(example.shape, target.shape)

        processed_example = example.unsqueeze(0), target
        return processed_example
