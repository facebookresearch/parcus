# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from utils.bert import load_bert
from pathlib import Path
import torch
import os
import numpy as np
import pickle
from gensim.test.utils import datapath
import gensim.models.fasttext


class HatespeechInputFeatures(object):
    """A single set of features of data."""

    def __init__(self, unique_example_id, unique_sentence_id, tokens, annotations, input_ids, input_mask, input_type_ids):
        self.unique_example_id = unique_example_id,
        self.unique_sentence_id = unique_sentence_id,
        self.tokens = tokens
        self.annotations = annotations
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.input_type_ids = input_type_ids


class HatespeechBuilder():

    def __init__(self, formatted_data_folder, processed_dataset_folder):
        self.formatted_data_folder = formatted_data_folder
        self.processed_dataset_folder = processed_dataset_folder

    def compute_hatespeech_embeddings(self, gensim_model, device):

        for dataset_type in ['train', 'test']:

            with open(Path(self.processed_dataset_folder, f'formatted_{dataset_type}_dataset_fasttext.pickle'), 'rb') as f:
                dataset = pickle.load(f)
            no_examples = len(dataset)

            dataset_path = Path(self.processed_dataset_folder, f'hatespeech_{dataset_type}_fasttext')
            if not os.path.exists(dataset_path):
                os.makedirs(dataset_path)

            example_id = 0
            for target, highlight, example in dataset:

                example_filename = Path(dataset_path, f'example_{example_id}.torch')
                example_sentences = {'target': target, 'sentences': [], 'highlighted': highlight}

                for sentence in example:
                    # Extraxt fields of the feature i.e. the sentence
                    unique_example_id = sentence.unique_example_id[0]  # tuple of single element
                    unique_sentence_id = sentence.unique_sentence_id[0]  # tuple of single element
                    tokens = sentence.tokens

                    if len(tokens) == 0:  # Empty sentence
                        continue

                    tokens = tokens

                    annotations = sentence.annotations

                    assert annotations is not None
                    assert len(annotations) != 0

                    tokens_embeddings = torch.tensor([gensim_model[t] for t in tokens])

                    tokens_mean = torch.sum(tokens_embeddings, dim=0) / (tokens_embeddings.shape[0])
                    # Combine the sentences in a single dictionary and save as a torch file
                    sentence_dict = {'unique_sentence_id': sentence.unique_sentence_id,
                                     'example_id': example_id,
                                     'tokens_annotations': annotations,
                                     'tokens_embeddings': tokens_embeddings,
                                     'sentence_embeddings': tokens_mean.unsqueeze(0),
                                     'tokens': tokens}

                    assert len(sentence_dict['tokens_annotations']) == int(sentence_dict['tokens_embeddings'].shape[0]), (len(sentence_dict['tokens_annotations']), sentence_dict['tokens_embeddings'].shape[0])

                    example_sentences['sentences'].append(sentence_dict)

                # Storing example dict in a torch file
                torch.save(example_sentences, example_filename)

                print(f'Completed example {example_id + 1}/{no_examples}')
                example_id += 1
        print('')  # just add a newline between training, validation and test


    def process_embeddings(self, embedding_dim):
        for dataset_type in ['train', 'test']:

            dataset_path = Path(self.processed_dataset_folder, f'hatespeech_{dataset_type}_fasttext')
            processed_path = Path(self.processed_dataset_folder, f'hatespeech_{dataset_type}_fasttext', 'processed')

            example_no = 0
            for filename in [f for f in sorted(os.listdir(dataset_path)) if '.torch' in f]:
                example_no += 1
                example_filename = filename
                example = torch.load(Path(dataset_path, example_filename))
                target, highlighted, example = example['target'], example['highlighted'], example['sentences']

                processed_example = {'target': torch.tensor([target], dtype=torch.long), 'tokens': [],
                                     'tokens_embeddings': None, 'sentence_embeddings': None,
                                     'tokens_annotations': None}

                # This will be needed to compute a single indexing for all tokens in the DOCUMENT
                starting_token_idx = 0
                sentence_idx = -1  # used by reference embeddings

                for sentence in example:
                    sentence_idx += 1
                    unique_sentence_id = sentence['unique_sentence_id']
                    sentence_example_id = sentence['example_id']

                    # The baseline will take the mean of the embeddings at runtime!
                    tokens_annotations = torch.from_numpy(np.array(sentence['tokens_annotations'])).long()  # CLS and SEP already removed
                    tokens_embeddings = sentence['tokens_embeddings']  # CLS and SEP already removed
                    sentence_embeddings = sentence['sentence_embeddings']  # CLS and SEP already removed
                    sentence_tokens = sentence['tokens']  # CLS and SEP already removed

                    # print(tokens_annotations.shape, tokens_embeddings.shape, sentence_embeddings.shape, len(sentence_tokens))

                    # Construct ordered pairs of tokens (all of them for now)
                    no_tokens = len(sentence_tokens)

                    # Now update example info by concatenating everything
                    for key, val in [('tokens_embeddings', tokens_embeddings),
                                     ('tokens_annotations', tokens_annotations),
                                     ('sentence_embeddings', sentence_embeddings)]:

                        if processed_example[key] is None:
                            processed_example[key] = val
                        else:
                            processed_example[key] = torch.cat((processed_example[key], val), dim=0)

                    starting_token_idx += no_tokens

                    processed_example['tokens'].extend(sentence_tokens)

                if highlighted == 1:
                    store_path = Path(processed_path, 'highlighted')
                else:
                    store_path = processed_path

                if not os.path.exists(store_path):
                    os.makedirs(store_path)

                if f'{example_filename[:-6]}_processed.torch' == 'example_1139_processed.torch':
                    print(example)
                    print(processed_example)

                torch.save(processed_example, Path(store_path, f'{example_filename[:-6]}_processed.torch'))

                if example_no % 1000 == 0:
                    print(f'Processed {example_no} examples')


if __name__ == '__main__':
    os.environ['OMP_NUM_THREADS'] = "1"

    formatted_data_folder = '../../../data/hatespeech'
    processed_data_folder = '../../../data/hatespeech'

    gensim_hatespeech_parser = HatespeechBuilder(formatted_data_folder, processed_data_folder)

    cap_path = "../../../data/fasttextmodel/wiki.en.bin"
    gensim_model = gensim.models.fasttext.load_facebook_vectors(cap_path)

    # Computes embeddings for each example
    gensim_hatespeech_parser.compute_hatespeech_embeddings(gensim_model, 'cpu')

    # Processes examples for subsequent training
    gensim_hatespeech_parser.process_embeddings(embedding_dim=300)
