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
import pandas as pd

# An example may be composed by multiple Input Features (i.e. sentences)
from utils.bert import load_bert
from utils.spacy import load_spacy


# An example may be composed by multiple Input Features (i.e. sentences)
class SpouseInputFeatures(object):
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
class SpouseInputExample(object):

    def __init__(self, unique_id, highlight, target, text_a, text_b, candidate1, candidate2):
        self.unique_id = unique_id
        self.target = target
        self.text_a = text_a
        self.text_b = text_b
        self.highlight = highlight
        self.candidate1 = candidate1
        self.candidate2 = candidate2


def fix_documents(pipeline, data_folder):

    # Try to improve the dataset which has clear lacks of punctuation, so BERT will not work properly on many cases

    with open(Path(data_folder, f'spouse_train_set.json'), 'r') as f:
        train_set = json.load(f)

    with open(Path(data_folder, f'spouse_validation_set.json'), 'r') as f:
        validation_set = json.load(f)

    with open(Path(data_folder, f'spouse_test_set.json'), 'r') as f:
        test_set = json.load(f)

    for dataset, dataset_type in [(train_set, 'train'), (validation_set, 'validation'), (test_set, 'test')]:
        print(f'Parsing {dataset_type} set')

        for pos_neg in ['positive', 'negative']:
            for i, doc in enumerate(dataset[pos_neg]):
                if (i+1) % 500 == 0:
                    print(f'example {i+1} of {pos_neg} samples')

                # DO NOT USE PARSED, as it contains ALEX and CHRIS
                raw = doc['original']

                # Try to split document in multiple sentences.
                # In most cases there is a high number of spaces before a capital letter. Use a full stop there
                prepr = re.sub(r'\s\s[\s]+', '. ', raw)
                prepr = re.sub(r'\.[\.]+', '.', prepr)

                parsed = pipeline(prepr)

                new_example = [t.text for t in parsed]

                doc['parsed'] = new_example

        with open(Path(data_folder, f'spouse_{dataset_type}_set_finetune.json'), 'w') as f:
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
                        parsed_example.append(token)
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
                        parsed_example.append('</pos>')
                        possible_match = False
                        left_out_tokens = []
                    elif left_out_tokens == ['<', 'po', '##s'] and token == '>':
                        parsed_example.append('<pos>')
                        possible_match = False
                        left_out_tokens = []
                    elif left_out_tokens == ['<', '/', 'ne', '##g'] and token == '>':
                        parsed_example.append('</neg>')
                        possible_match = False
                        left_out_tokens = []
                    elif left_out_tokens == ['<', 'ne', '##g'] and token == '>':
                        parsed_example.append('<neg>')
                        possible_match = False
                        left_out_tokens = []
                    else:
                        parsed_example.extend([t for t in left_out_tokens])
                        possible_match = False
                        left_out_tokens = []

            # ----------------- End of finite state machine ------------------ #

            # Account for [CLS] and [SEP] with "- 2" and some more tokens (upper bound) due to candidates


        #print(len(parsed_example))
        if len(parsed_example) > seq_length - 2 - 20:
            parsed_example = parsed_example[0:(seq_length - 2 - 20)]



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

        input_type_ids = []
        annotations = []
        tokens = []

        tokens.append("[CLS]")
        input_type_ids.append(0)
        annotations.append(0)

        for token in parsed_example:
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
                input_type_ids.append(0)

        tokens.append("[SEP]")
        input_type_ids.append(0)
        annotations.append(0)

        for token in tokenizer.tokenize(example.candidate1):
            tokens.append(token)
            input_type_ids.append(0)
            annotations.append(0)

        tokens.append("[SEP]")
        input_type_ids.append(0)
        annotations.append(0)

        for token in tokenizer.tokenize(example.candidate2):
            tokens.append(token)
            input_type_ids.append(0)
            annotations.append(0)

        tokens.append("[SEP]")
        input_type_ids.append(0)
        annotations.append(0)

        # we also create a sentence ID, it may be useful
        sentence_unique_id = example_unique_id

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

        # print(len(input_ids), len(input_mask), len(input_type_ids))

        assert len(input_ids) == seq_length, (len(input_ids), seq_length)
        assert len(input_mask) == seq_length, (len(input_mask), seq_length)
        assert len(input_type_ids) == seq_length, (len(input_type_ids), seq_length)
        assert len(tokens) == len(annotations), (len(tokens), len(annotations))

        # print(f'Sentence unique id is {sentence_unique_id}')
        example_features.append(
            SpouseInputFeatures(
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
    for pos_neg in ['positive', 'negative']:
        for doc in dataset[pos_neg]:
            i += 1
            if i % 2000 == 0:
                print(i, end='')
                print('\r', end='')

            sample = doc['parsed']
            target = doc['target']
            c1 = doc['candidate-1']
            c2 = doc['candidate-2']
            highlight = doc['highlighted']
            text_a = sample  # this is a list of sentences
            text_b = None  # we do not have pairs of sentences, we just need words embeddings for each document

            examples.append(
                SpouseInputExample(unique_id=unique_id, target=target, highlight=highlight, text_a=text_a, text_b=text_b,
                                   candidate1=c1, candidate2=c2))
            unique_id += 1
            # See convert_examples_to_features
        print('')

    return examples


def compute_bert_formatted_inputs(data_folder, max_seq_length=512):

    tokenizer, _, _, _ = load_bert(lower_case=True, base_version=True)

    # Generate formatted input
    for dataset_type in ['train', 'validation', 'test']:
        with open(Path(data_folder, f'spouse_{dataset_type}_set_finetune.json'), 'r') as f:
            dataset = json.load(f)

            # Just convert examples to InputExamples
            splitted_dataset = _process_examples(dataset)

            # Convert to BERT input
            examples_features = _convert_examples_to_features(
                examples=splitted_dataset, seq_length=max_seq_length, tokenizer=tokenizer)

            print(f'Len of parsed examples is {len(examples_features)}, just to double check')

            with open(Path(data_folder, f'spouse_formatted_{dataset_type}_finetune_dataset.pickle'), 'wb') as w:
                pickle.dump(examples_features, w)


def main():
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    os.environ['OMP_NUM_THREADS'] = "1"

    data_folder = '../../../data/Spouse'

    fix_documents(load_spacy(), data_folder)

    compute_bert_formatted_inputs(data_folder)


if __name__ == '__main__':
    main()