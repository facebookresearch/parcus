# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from random import shuffle

import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch.utils.data.dataset import ConcatDataset
import torch.nn.functional as F
from datasets.utils import custom_collate


def process_outputs(predictions, no_rules, threshold):
    no_elements = predictions.shape[0] // no_rules
    Ls_set = None

    for i in range(no_rules):

        softmax_preds = F.softmax(predictions[no_elements * i:no_elements * (i + 1)], dim=1)

        mat = torch.zeros(softmax_preds.shape[0], dtype=torch.long)
        mat[softmax_preds[:, 1] >= 0.5 + threshold] = 1
        mat[softmax_preds[:, 0] >= 0.5 + threshold] = 2
        mat = mat.unsqueeze(1)

        if Ls_set is None:
            Ls_set = mat
        else:
            Ls_set = torch.cat((Ls_set, mat), dim=1)

    return Ls_set

def _compute_all_targets(dataset, batch_size):
    # shuffle MUST stay false
    loader = DataLoader(dataset, batch_size=batch_size, collate_fn=custom_collate, shuffle=False)

    all_targets = None

    for _, targets in loader:

        targets, _ = targets

        if all_targets is None:
            all_targets = np.expand_dims(targets, 1)
        else:
            all_targets = np.concatenate((all_targets, np.expand_dims(targets, 1)), axis=0)

    return all_targets


def diff(first, second):
    second = set(second)
    return [item for item in first if item not in second]


def get_data_splits(highlighted_dataset, non_highlighted_dataset, train_size, target_idx):

    train_idxs = list(range(len(highlighted_dataset)))
    shuffle(train_idxs)

    other_idxs = [i+len(highlighted_dataset) for i in range(len(non_highlighted_dataset))]
    shuffle(other_idxs)

    concatenated = ConcatDataset((highlighted_dataset, non_highlighted_dataset))
    concatenated_idxs = train_idxs + other_idxs

    # WARNING: HateSpeech highlighted samples are all POSITIVE!
    # I am putting all of them at the beginning, so that they are always chosen
    # Already shuffled, I do not need this thing

    # Compute train split

    train_split = []
    no_pos, no_neg = 0, 0
    for i in concatenated_idxs:

        if concatenated[i][target_idx] == 1:
            if no_pos < (train_size // 10) * 5:
                train_split.append(i)
                no_pos += 1
        elif concatenated[i][target_idx] != 1:
            if no_neg < (train_size // 10) * 5:
                train_split.append(i)
                no_neg += 1

        if len(train_split) == train_size:
            break

    concatenated_idxs_no_train = diff(concatenated_idxs, train_split)
    shuffle(concatenated_idxs_no_train)

    valid_split = []
    no_pos, no_neg = 0, 0
    for i in concatenated_idxs_no_train:

        # Valid size should be the same as train size
        if concatenated[i][target_idx] == 1:
            if no_pos < (train_size // 10) * 5:
                valid_split.append(i)
                no_pos += 1
        elif concatenated[i][target_idx] != 1:
            if no_neg < (train_size // 10) * 5:
                valid_split.append(i)
                no_neg += 1

        if len(valid_split) == train_size:
            break

    # Compute difference to get the remaining data points as the validation set
    unlabelled_split = diff(concatenated_idxs_no_train, valid_split)

    assert not (set(train_split) & set(valid_split) & set(unlabelled_split))

    print(train_split, valid_split)
    print(len(train_split), len(valid_split), len(unlabelled_split))
    return concatenated, train_split, valid_split, unlabelled_split
