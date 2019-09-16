# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import os

import torch
from pathlib import Path
from torch.utils.data.dataset import Dataset


class NREHatespeechDataset(Dataset):

    def __init__(self, samples_folder, cache_data=True, rationale_noise=0.):
        print(samples_folder)
        self.rationale_noise = rationale_noise
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
            annotations = example['tokens_annotations'].unsqueeze(1)
            if annotations is not None:
                rnd_noise = rnd_noise = (
                            torch.rand(annotations.shape, dtype=torch.float) > (1 - self.rationale_noise)).long()
                annotations = annotations + rnd_noise
                annotations[annotations == 2] = 1
        else:
            if self.cached[idx] is None:
                example = torch.load(Path(self.folder, self.files[idx]))
                annotations = example['tokens_annotations'].unsqueeze(1)
                if annotations is not None:
                    rnd_noise = rnd_noise = (
                                torch.rand(annotations.shape, dtype=torch.float) > (1 - self.rationale_noise)).long()
                    annotations = annotations + rnd_noise
                    annotations[annotations == 2] = 1
                example['tokens_annotations'] = annotations

                self.cached[idx] = example
            else:
                example = self.cached[idx]


        annotations = example['tokens_annotations']
        tokens_embeddings = example['tokens_embeddings']
        embeddings = tokens_embeddings  # ?x768

        mask_input = torch.ones(embeddings.shape[0])

        # convert -1 and +1 in 0, 1
        target = (example['target'] + 1) / 2

        processed_example = embeddings, annotations, mask_input, target, torch.tensor([idx]).long()

        return processed_example


class NREMovieReviewDataset(Dataset):

    def __init__(self, samples_folder, cache_data=True, rationale_noise=0.):
        print(samples_folder)
        self.rationale_noise = rationale_noise

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
            annotations = example['tokens_annotations'].unsqueeze(1)
            if annotations is not None:
                rnd_noise = rnd_noise = (
                            torch.rand(annotations.shape, dtype=torch.float) > (1 - self.rationale_noise)).long()
                annotations = annotations + rnd_noise
                annotations[annotations == 2] = 1
        else:
            if self.cached[idx] is None:
                example = torch.load(Path(self.folder, self.files[idx]))
                annotations = example['tokens_annotations'].unsqueeze(1)
                if annotations is not None:
                    rnd_noise = rnd_noise = (
                                torch.rand(annotations.shape, dtype=torch.float) > (1 - self.rationale_noise)).long()
                    annotations = annotations + rnd_noise
                    annotations[annotations == 2] = 1
                example['tokens_annotations'] = annotations

                self.cached[idx] = example
            else:
                example = self.cached[idx]


        annotations = example['tokens_annotations']
        tokens_embeddings = example['tokens_embeddings']
        embeddings = tokens_embeddings  # ?x768

        mask_input = torch.ones(embeddings.shape[0])

        # convert -1 and +1 in 0, 1
        target = (example['target'] + 1) / 2

        processed_example = embeddings, annotations, mask_input, target, torch.tensor([idx]).long()

        return processed_example


class NRESpouseDataset(Dataset):

    def __init__(self, samples_folder, cache_data=True, rationale_noise=0.):
        print(samples_folder)
        self.rationale_noise = rationale_noise
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
            annotations = example['tokens_annotations'].unsqueeze(1)
            if annotations is not None:
                rnd_noise = rnd_noise = (
                            torch.rand(annotations.shape, dtype=torch.float) > (1 - self.rationale_noise)).long()
                annotations = annotations + rnd_noise
                annotations[annotations == 2] = 1
        else:
            if self.cached[idx] is None:
                example = torch.load(Path(self.folder, self.files[idx]))
                annotations = example['tokens_annotations'].unsqueeze(1)
                if annotations is not None:
                    rnd_noise = rnd_noise = (
                                torch.rand(annotations.shape, dtype=torch.float) > (1 - self.rationale_noise)).long()
                    annotations = annotations + rnd_noise
                    annotations[annotations == 2] = 1
                example['tokens_annotations'] = annotations

                self.cached[idx] = example
            else:
                example = self.cached[idx]


        annotations = example['tokens_annotations']
        tokens_embeddings = example['tokens_embeddings']
        both_present = example['tokens_both_present']

        alex_chris_mask = example['alex_chris_mask']

        # Used at inference time
        mask_input = both_present

        # Mask alex and chris embeddings
        embeddings = torch.mul(tokens_embeddings, alex_chris_mask.unsqueeze(1))  # ?x768

        # convert -1 and +1 in 0, 1
        target = (example['target'] + 1) / 2

        processed_example = embeddings, annotations, mask_input, target, torch.tensor([idx]).long()

        return processed_example
