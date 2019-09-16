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
from random import shuffle
from pathlib import Path


# An example may be composed by multiple Input Features (i.e. sentences)
import torch

from utils.bert import load_bert
from utils.spacy import load_spacy


# Code adapted from pytorch BERT repository
def _convert_examples_to_features(data_folder, examples, pipeline):
    """Loads a data file into a list of `InputFeature`s."""

    dataset_features = []

    with open(Path(data_folder, f'word_to_idx.json'), 'r', encoding='utf-8') as f:
        word_to_idx = json.load(f)

    no_ngrams = len(word_to_idx.keys())

    idx = 0
    for i, example in enumerate(examples):

        idx += 1
        if idx % 100 == 0:
            print(f'Parsing sample no {idx}', end='')
            print('\r', end='')

        ngram_feats = torch.zeros(no_ngrams + 1)  # Add length of tweet as well
        targets = torch.zeros(1)  # Add length of tweet as well

        tweet = ' '.join(example['sample'])

        # Remove links and highlights from the tweet
        tweet = re.sub(
            r'''(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'".,<>?«»“”‘’]))''',
            " ", tweet)

        tweet = re.sub('<POS>', "", tweet)
        tweet = re.sub('<NEG>', "", tweet)
        tweet = re.sub('</POS>', "", tweet)
        tweet = re.sub('</NEG>', "", tweet)
        tweet = re.sub('\n', "", tweet)
        tweet = re.sub('   ', " ", tweet)

        doc = pipeline(tweet)

        tweet_length = len(doc)

        # Remove stop words (follow work from Waseem and Hovy)
        tokens = [token.text.strip() for token in doc if not token.is_stop]

        n = 1
        unigrams = set([(w[i:i + n]).lower() for w in tokens for i in range(len(w) - n + 1)])

        n = 2
        bigrams = set([(w[i:i + n]).lower() for w in tokens for i in range(len(w) - n + 1)])

        n = 3
        trigrams = set([(w[i:i + n]).lower() for w in tokens for i in range(len(w) - n + 1)])

        n = 4
        fourgrams = set([(w[i:i + n]).lower() for w in tokens for i in range(len(w) - n + 1)])

        idxs = list(set([word_to_idx[gram] for gram in (unigrams.union(bigrams, trigrams, fourgrams))]))

        ngram_feats[idxs] = 1
        ngram_feats[-1] = tweet_length
        targets[0] = example['target']

        dataset_features.append((ngram_feats, example['highlighted'], targets))
    return dataset_features


def compute_bert_formatted_inputs(data_folder, pipeline):

    # Generate formatted input
    for dataset_type in ['train', 'test']:
        with open(Path(data_folder, f'splitted_{dataset_type}_sentences.json'), 'r') as f:
            dataset = json.load(f)

            examples_features = _convert_examples_to_features(data_folder, examples=dataset, pipeline=pipeline)

            print(f'Len of parsed examples is {len(examples_features)}, just to double check')

            with open(Path(data_folder, f'formatted_ngrams_{dataset_type}_dataset.pickle'), 'wb') as w:
                pickle.dump(examples_features, w)

def main():
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    os.environ['OMP_NUM_THREADS'] = "1"

    # NOTE This uses splitted_{dataset} computed by Hatespeech_Preprocess.py (parse_tweets function)

    data_folder = '../../../data/hatespeech'

    pipeline = load_spacy()

    compute_bert_formatted_inputs(data_folder, pipeline)


if __name__ == '__main__':
    main()