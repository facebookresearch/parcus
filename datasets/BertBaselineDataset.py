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

class BertBaselineDataset(Dataset):

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

        sentences = example['sentence_embeddings']

        doc_embedding = None

        for embedding in sentences:

            #print(embedding.shape)

            # print(s['tokens'])
            if doc_embedding is None:
                doc_embedding = embedding
            else:
                doc_embedding += embedding

        doc_embedding = doc_embedding / len(sentences)

        # convert -1 and +1 in 0, 1
        target = (example['target'] + 1) / 2
        processed_example = doc_embedding.squeeze(), target

        return processed_example


class BertBaselineTokensDataset(Dataset):

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

        tokens_embeddings = example['tokens_embeddings']
        annotations = example['tokens_annotations'].unsqueeze(1)

        embeddings = tokens_embeddings  # ?x768

        mask_input = torch.ones(embeddings.shape[0])

        # convert -1 and +1 in 0, 1
        target = (example['target'] + 1) / 2

        processed_example = embeddings, annotations, mask_input, target, torch.tensor([idx]).long()

        return processed_example
