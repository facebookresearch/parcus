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


def load_tweets(data_folder):
    tweets = []
    for line in open(Path(data_folder, 'amateur_expert.json'), 'r'):
        tweets.append(json.loads(line))

    racism = []
    neither = []
    sexism = []

    for i, tweet in enumerate(tweets):
        label = tweets[i]["Annotation"]
        text = tweets[i]["text"]
        if label == "Sexism":
            sexism.append(text)
        if label == "Racism":
            racism.append(text)
        if label == "Neither":
            neither.append(text)

    return racism, sexism, neither


def train_test_tweets(pos, neg):
    train_dataset = []
    test_dataset = []  # should contain last 100 examples of all folders

    shuffle(pos)
    shuffle(neg)

    for i, tweet in enumerate(pos):
        target = 1
        if i < len(pos) // 2:
            train_dataset.append({'sample': tweet, 'target': target, 'highlighted': 0})
        else:
            test_dataset.append({'sample': tweet, 'target': target, 'highlighted': 0})

    for i, tweet in enumerate(neg):
        target = -1
        if i < len(neg) // 2:
            train_dataset.append({'sample': tweet, 'target': target, 'highlighted': 0})
        else:
            test_dataset.append({'sample': tweet, 'target': target, 'highlighted': 0})

    return train_dataset, test_dataset


def parse_tweets(pipeline, train_dataset, test_dataset, data_folder):

    for dataset_type, dataset in [('train', train_dataset), ('test', test_dataset)]:
        idx = 0
        for example in dataset:
            idx += 1
            if idx % 100 == 0:
                print(f'Parsing sample no {idx}', end='')
                print('\r', end='')

            doc = example['sample']

            doc = pipeline(doc)

            new_example = []
            first_sentence = True
            sentence = []

            for token in doc:
                if token.is_sent_start and first_sentence:
                    sentence.append(token.text)
                    first_sentence = False

                elif token.is_sent_start and not first_sentence:
                    new_example.append(" ".join(sentence))
                    sentence = []
                    sentence.append(token.text)
                else:
                    sentence.append(token.text)

            new_example.append(" ".join(sentence))
            example['sample'] = new_example

        print('')

        with open(Path(data_folder, f'splitted_{dataset_type}_sentences.json'), 'w', encoding='utf-8') as f:
            json.dump(dataset, f)


# Code adapted from pytorch BERT repository
def _convert_examples_to_features(examples, seq_length, tokenizer):
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
            tokens = tokenizer.tokenize(sentence)

            tokens_sentence = []  # the tokens with "corrected" annotation placeholders

            # ------ Finite State Machine to replace specific substrings  ------ #
            left_out_tokens = []
            possible_match = False

            for token in tokens:
                if not possible_match:
                    if token == '<':
                        possible_match = True  # start tracking possible tag
                        left_out_tokens.append(token)
                    else:
                        tokens_sentence.append(token)
                else:
                    if left_out_tokens == ['<'] and token in ['/', 'ne', 'po'] or \
                            left_out_tokens == ['<', '/'] and token in ['ne'] or \
                            left_out_tokens == ['<', '/'] and token in ['po'] or \
                            left_out_tokens == ['<', 'po'] and token in ['##s'] or \
                            left_out_tokens == ['<', 'ne'] and token in ['##g'] or \
                            left_out_tokens == ['<', '/', 'po'] and token in ['##s'] or \
                            left_out_tokens == ['<', '/', 'ne'] and token in ['##g']:
                        left_out_tokens.append(token)
                    elif left_out_tokens == ['<', '/', 'po', '##s'] and token == '>':
                        tokens_sentence.append('</pos>')
                        possible_match = False
                        left_out_tokens = []
                    elif left_out_tokens == ['<', 'po', '##s'] and token == '>':
                        tokens_sentence.append('<pos>')
                        possible_match = False
                        left_out_tokens = []
                    elif left_out_tokens == ['<', '/', 'ne', '##g'] and token == '>':
                        tokens_sentence.append('</neg>')
                        possible_match = False
                        left_out_tokens = []
                    elif left_out_tokens == ['<', 'ne', '##g'] and token == '>':
                        tokens_sentence.append('<neg>')
                        possible_match = False
                        left_out_tokens = []
                    else:
                        tokens_sentence.extend([t for t in left_out_tokens])
                        possible_match = False
                        left_out_tokens = []

            # ----------------- End of finite state machine ------------------ #

            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_sentence) > seq_length - 2:
                tokens_sentence = tokens_sentence[0:(seq_length - 2)]

            # Append parsed sentence to the parsed example (document)
            parsed_example.append([t for t in tokens_sentence])

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids:   0   0  0    0    0     0      0   0    1  1  1   1  1   1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids:   0   0   0   0  0     0   0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambigiously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.

        # We now prepare the data for BERT

        annotate_as_neg = False  # Needed to associate an annotation to each token
        annotate_as_pos = False  # Needed to associate an annotation to each token

        sentences = []
        for sentence in parsed_example:
            input_type_ids = []
            annotations = []
            tokens = []

            tokens.append("[CLS]")
            input_type_ids.append(0)
            annotations.append(0)

            for token in sentence:
                if token == '<neg>':
                    # print(f'found {token}!')
                    assert not annotate_as_pos
                    annotate_as_neg = True
                elif token == '<pos>':
                    # print(f'found {token}!')
                    assert not annotate_as_neg
                    annotate_as_pos = True
                elif token == '</neg>':
                    # print(f'found {token}!')
                    assert annotate_as_neg
                    assert not annotate_as_pos
                    annotate_as_neg = False
                elif token == '</pos>':
                    # print(f'found {token}!')
                    assert annotate_as_pos, sentence
                    assert not annotate_as_neg
                    annotate_as_pos = False
                else:
                    if annotate_as_neg or annotate_as_pos:
                        annotations.append(1)
                    else:
                        annotations.append(0)
                    tokens.append(token)
                    input_type_ids.append(0)

            tokens.append("[SEP]")
            input_type_ids.append(0)
            annotations.append(0)

            sentences.append((tokens, annotations, input_type_ids))

        # Now it is time to store things

        # we also create a sentence ID, it may be useful
        sentence_unique_id = example_unique_id

        for tokens, annotations, input_type_ids in sentences:

            # THIS CREATES THE BERT REAL INPUT
            input_ids = tokenizer.convert_tokens_to_ids(tokens)

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            input_mask = [1] * len(input_ids)

            # Zero-pad up to the sequence length.
            while len(input_ids) < seq_length:
                input_ids.append(0)
                input_mask.append(0)
                input_type_ids.append(0)

            assert len(input_ids) == seq_length
            assert len(input_mask) == seq_length
            assert len(input_type_ids) == seq_length

            # print(f'Sentence unique id is {sentence_unique_id}')
            example_features.append(
                HatespeechInputFeatures(
                    unique_example_id=example_unique_id,
                    unique_sentence_id=sentence_unique_id,
                    tokens=tokens,
                    annotations=annotations,
                    input_ids=input_ids,
                    input_mask=input_mask,
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
        target = doc['target']
        highlight = doc['highlighted']
        text_a = sample  # this is a list of sentences
        text_b = None  # we do not have pairs of sentences, we just need words embeddings for each document

        examples.append(
            HatespeechInputExample(unique_id=unique_id, target=target, highlight=highlight, text_a=text_a, text_b=text_b))
        unique_id += 1
        # See convert_examples_to_features

    return examples


def compute_bert_formatted_inputs(data_folder, max_seq_length=128):

    tokenizer, _, _, _ = load_bert(lower_case=True, base_version=True)

    # Generate formatted input
    for dataset_type in ['train', 'test']:
        with open(Path(data_folder, f'splitted_{dataset_type}_sentences.json'), 'r') as f:
            dataset = json.load(f)

            # Just convert examples to InputExamples
            splitted_dataset = _process_examples(dataset)

            # Convert to BERT input
            # examples_features is a list of list. The outer list holds examples (documents), the inner list holds one feature object per sentence
            examples_features = _convert_examples_to_features(
                examples=splitted_dataset, seq_length=max_seq_length, tokenizer=tokenizer)

            print(f'Len of parsed examples is {len(examples_features)}, just to double check')

            with open(Path(data_folder, f'formatted_{dataset_type}_dataset.pickle'), 'wb') as w:
                pickle.dump(examples_features, w)


def main():

    data_folder = '../../../data/hatespeech'

    racism, sexism, neither = load_tweets(data_folder)
    print(f'No. sexist tweets: {len(sexism)} \t No. racist tweets: {len(racism)} \t No. other tweets: {len(neither)}')
    print(f'Total tweets: {len(sexism) + len(racism) + len(neither)}')

    # Merge racism with sexism
    pos = racism + sexism
    neg = neither

    train, test = train_test_tweets(pos, neg)

    pipeline = load_spacy()

    parse_tweets(pipeline, train, test, data_folder)

    compute_bert_formatted_inputs(data_folder)


if __name__ == '__main__':
    main()