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



class BertFinetuneDataset(Dataset):

    def __init__(self, samples_folder, cache_data=True):
        self.folder = samples_folder
        self.cache_data = cache_data
        self.files = [filename for filename in sorted(os.listdir(samples_folder)) if '.torch' in filename]
        self.cached = [None for _ in range(len(self.files))]

    def __len__(self):
        # no of examples
        return len(self.files)

    def get_targets(self, idx):
        return torch.load(Path(self.folder, self.files[idx]))['target'].numpy()

    def __getitem__(self, idx):

        if not self.cache_data:
            example = torch.load(Path(self.folder, self.files[idx]))
        else:
            if self.cached[idx] is None:
                example = torch.load(Path(self.folder, self.files[idx]))
                self.cached[idx] = example
            else:
                example = self.cached[idx]

        tokens_ids = example['input_ids']
        annotations = example['tokens_annotations'].unsqueeze(1)
        mask_input = example['input_mask']

        # convert -1 and +1 in 0, 1
        target = (example['target'] + 1) / 2

        processed_example = tokens_ids, annotations, mask_input, target, torch.tensor([idx]).long()

        return processed_example
