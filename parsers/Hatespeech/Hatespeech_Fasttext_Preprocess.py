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
from utils.bert import load_bert
from utils.spacy import load_spacy


class HatespeechInputFeatures(object):
    """A single set of features of data."""

    def __init__(self, unique_example_id, unique_sentence_id, tokens, annotations, input_ids, input_mask, input_type_ids):
        self.unique_example_id=unique_example_id,
        self.unique_sentence_id=unique_sentence_id,
        self.tokens = tokens
        self.annotations = annotations
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.input_type_ids = input_type_ids


# Defines a single Input Example
class HatespeechInputExample(object):

    def __init__(self, unique_id, highlight, target, text_a, text_b):
        self.unique_id = unique_id
        self.target = target
        self.text_a = text_a
        self.text_b = text_b
        self.highlight = highlight

# Code adapted from pytorch BERT repository
def _convert_examples_to_features(examples, seq_length):
    """Loads a data file into a list of `InputFeature`s."""

    # This variable holds a list of examples. Each example is a list of sentences in the form of "features"
    dataset_features = []
    for example in examples:
        # get example unique ID
        example_unique_id = example.unique_id

        example_highlight = example.highlight

        # get target label associated to the document
        example_target = example.target

        # get the sentences
        sentences = example.text_a  # text_b always None

        # Remove links from the tweet
        sentences = [re.sub(
            r'''(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'".,<>?«»“”‘’]))''',
            " ", s) for s in sentences]

        # istantiate a list of features, one per sentence
        example_features = []

        # The parsed sentence with <pos> and <neg> tags recombined
        parsed_example = []

        for sentence in sentences:
            tokens = sentence.split()
            # Append parsed sentence to the parsed example (document)
            parsed_example.append([t.lower() for t in tokens])

        # We now prepare the data for BERT

        annotate_as_neg = False  # Needed to associate an annotation to each token
        annotate_as_pos = False  # Needed to associate an annotation to each token

        sentences = []
        for sentence in parsed_example:

            if len(sentence) == 0 or sentence[0] == '':
                continue

            input_type_ids = []
            annotations = []
            tokens = []

            for token in sentence:

                if token == '<neg>':
                    #print(f'found {token}!')
                    assert not annotate_as_pos
                    annotate_as_neg = True
                elif token == '<pos>':
                    #print(f'found {token}!')
                    assert not annotate_as_neg
                    annotate_as_pos = True
                elif token == '</neg>':
                    #print(f'found {token}!')
                    assert annotate_as_neg
                    assert not annotate_as_pos
                    annotate_as_neg = False
                elif token == '</pos>':
                    #print(f'found {token}!')
                    assert annotate_as_pos, sentence
                    assert not annotate_as_neg
                    annotate_as_pos = False
                else:
                    if annotate_as_neg or annotate_as_pos:
                        annotations.append(1)
                    else:
                        annotations.append(0)
                    tokens.append(token)

            assert len(tokens) != 0, example.text_a
            sentences.append((tokens, annotations, input_type_ids))

        # Now it is time to store things
        if len(sentences) == 0:
            continue

        # we also create a sentence ID, it may be useful
        sentence_unique_id = example_unique_id

        for tokens, annotations, input_type_ids in sentences:

            # print(f'Sentence unique id is {sentence_unique_id}')
            example_features.append(
                HatespeechInputFeatures(
                    unique_example_id=example_unique_id,
                    unique_sentence_id=sentence_unique_id,
                    tokens=tokens,
                    annotations=annotations,
                    input_ids=None,
                    input_mask=None,
                    input_type_ids=input_type_ids)
            )
            sentence_unique_id += 1

        dataset_features.append((example_target, example_highlight, example_features))
    return dataset_features


def _process_examples(dataset):
    """Read a list of `InputExample`s from list."""
    examples = []
    unique_id = 0

    i = 0
    for doc in dataset:
        i += 1
        if i % 2000 == 0:
            print(i)

        sample = doc['sample']

        assert len(sample) != 0

        target = doc['target']
        highlight = doc['highlighted']
        text_a = sample  # this is a list of sentences
        text_b = None  # we do not have pairs of sentences, we just need words embeddings for each document

        examples.append(
            HatespeechInputExample(unique_id=unique_id, target=target, highlight=highlight, text_a=text_a, text_b=text_b))
        unique_id += 1
        # See convert_examples_to_features

    return examples


def compute_fasttext_formatted_inputs(data_folder, max_seq_length=128):

    # Generate formatted input
    for dataset_type in ['train', 'test']:
        with open(Path(data_folder, f'splitted_{dataset_type}_sentences.json'), 'r') as f:
            dataset = json.load(f)

            # Just convert examples to InputExamples
            splitted_dataset = _process_examples(dataset)

            # Convert to BERT input
            # examples_features is a list of list. The outer list holds examples (documents), the inner list holds one feature object per sentence
            examples_features = _convert_examples_to_features(
                examples=splitted_dataset, seq_length=max_seq_length)

            print(f'Len of parsed examples is {len(examples_features)}, just to double check')

            with open(Path(data_folder, f'formatted_{dataset_type}_dataset_fasttext.pickle'), 'wb') as w:
                pickle.dump(examples_features, w)


def main():

    data_folder = '../../../data/hatespeech'

    compute_fasttext_formatted_inputs(data_folder)


if __name__ == '__main__':
    main()