# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import os
import re
import json
import pickle
import numpy as np
import torch
from random import shuffle
from pathlib import Path


# An example may be composed by multiple Input Features (i.e. sentences)
from utils.bert import load_bert
from utils.spacy import load_spacy


class HatespeechBuilder():

    def __init__(self, formatted_data_folder, processed_dataset_folder):
        self.formatted_data_folder = formatted_data_folder
        self.processed_dataset_folder = processed_dataset_folder

    def compute_hatespeech_embeddings(self):

        for dataset_type in ['test', 'train']:

            with open(Path(self.processed_dataset_folder, f'formatted_ngrams_{dataset_type}_dataset.pickle'), 'rb') as f:
                dataset = pickle.load(f)
            no_examples = len(dataset)

            dataset_path = Path(self.processed_dataset_folder, f'hatespeech_ngrams_{dataset_type}/processed')
            if not os.path.exists(dataset_path):
                os.makedirs(dataset_path)

            example_id = 0
            for ngram_feats, highlight, target in dataset:

                if highlight == 1:
                    store_path = Path(dataset_path, 'highlighted')
                else:
                    store_path = dataset_path

                if not os.path.exists(store_path):
                    os.makedirs(store_path)

                torch.save((ngram_feats, target), Path(store_path, f'example_{example_id}_processed.torch'))

                print(f'Completed example {example_id + 1}/{no_examples}')
                example_id += 1

        print('')  # just add a newline between training, validation and test


def main():
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    os.environ['OMP_NUM_THREADS'] = "1"

    formatted_data_folder = '../../../data/hatespeech'
    processed_data_folder = '../../../data/hatespeech'

    builder = HatespeechBuilder(formatted_data_folder, processed_data_folder)

    builder.compute_hatespeech_embeddings()


if __name__ == '__main__':
    main()