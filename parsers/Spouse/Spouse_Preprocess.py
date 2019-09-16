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

    def __init__(self, unique_id, highlight, target, text_a, text_b):
        self.unique_id = unique_id
        self.target = target
        self.text_a = text_a
        self.text_b = text_b
        self.highlight = highlight


def load_spouse(data_folder):
    spans_df = pd.read_csv(Path(data_folder, 'spouse_data/spans.csv'))
    spouse_df = pd.read_csv(Path(data_folder, 'spouse_data/spouse_table.csv'))

    train_dataset, validation_dataset, test_dataset = {'positive': [], 'negative': []}, {'positive': [],
                                                                                         'negative': []}, {
                                                          'positive': [], 'negative': []}

    spans_map = {}
    for row in spans_df.iterrows():
        spans_map[int(row[1]['id'])] = {'char_start': int(row[1]['char_start']),
                                        'char_end': int(row[1]['char_end']) + 1}

    i = 0
    for row in spouse_df.iterrows():
        i += 1
        sample = row[1]
        original_text = sample['text']
        text = sample['text']
        label = int(sample['label'])
        append_to = 'positive' if label == 1 else 'negative'

        split = int(sample['split'])
        person1 = sample['person1']
        person2 = sample['person2_id']

        char_start1, char_end1 = spans_map[person1]['char_start'], spans_map[person1]['char_end']
        char_start2, char_end2 = spans_map[person2]['char_start'], spans_map[person2]['char_end']

        candidate1, candidate2 = text[char_start1:char_end1], text[char_start2:char_end2]

        # First replace full names with Alice and Bob, respectively
        replace1 = "Alex"
        replace2 = "Chris"

        if candidate1[-1] == ' ':
            replace1 = replace1 + ' '

        if candidate2[-1] == ' ':
            replace2 = replace2 + ' '

        # I assume that, if the 's is present, the full name is used
        if candidate1[-2:] == '\'s':
            replace1 = replace1 + '\'s'
        if candidate2[-1] == '\'s':
            replace2 = replace2 + '\'s'

        text = text.replace(candidate1, replace1)
        text = text.replace(candidate2, replace2)

        # Now split, if any, name with surname and replace each with Alice or Bob.
        # This is because sometimes the full name is not used
        replace1 = "Alex"
        replace2 = "Chris"

        names1 = candidate1.split(' ')

        for name in names1:
            if name != '' and name[0].isupper():
                text = text.replace(name, ' ' + replace1 + ' ')

        names2 = candidate2.split(' ')
        for name in names2:
            if name != '' and name[0].isupper():
                text = text.replace(name, ' ' + replace2 + ' ')

        dict_to_append = {
            'parsed': text,
            'original': original_text,
            'candidate-1': candidate1,
            'candidate-2': candidate2,
            'target': label,
            'highlighted': 0
        }

        if split == 0:
            train_dataset[append_to].append(dict_to_append)
        elif split == 1:
            validation_dataset[append_to].append(dict_to_append)
        elif split == 2:
            test_dataset[append_to].append(dict_to_append)
        else:
            print("ERROR")

    for dataset, dataset_type in [(train_dataset, 'train'), (validation_dataset, 'validation'), (test_dataset, 'test')]:
        shuffle(dataset['positive'])
        shuffle(dataset['negative'])

        with open(Path(data_folder, f'spouse_{dataset_type}_set.json'), 'w') as f:
            json.dump(dataset, f)

    return train_dataset, validation_dataset, test_dataset


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

        sentences_length = []
        for pos_neg in ['positive', 'negative']:
            for i, doc in enumerate(dataset[pos_neg]):
                if (i+1) % 500 == 0:
                    print(f'example {i+1} of {pos_neg} samples')

                raw = doc['parsed'][0]

                # Try to split document in multiple sentences.
                # In most cases there is a high number of spaces before a capital letter. Use a full stop there
                prepr = re.sub(r'\s\s[\s]+', '. ', raw)
                prepr = re.sub(r'\.[\.]+', '.', prepr)

                parsed = pipeline(prepr)

                new_example = []
                first_sentence = True
                sentence = []

                for token in parsed:

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

                sentences_length.extend([len(s.split()) for s in new_example])

                doc['parsed'] = new_example

        with open(Path(data_folder, f'spouse_{dataset_type}_set.json'), 'w') as f:
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
            highlight = doc['highlighted']
            text_a = sample  # this is a list of sentences
            text_b = None  # we do not have pairs of sentences, we just need words embeddings for each document

            examples.append(
                SpouseInputExample(unique_id=unique_id, target=target, highlight=highlight, text_a=text_a, text_b=text_b))
            unique_id += 1
            # See convert_examples_to_features
        print('')

    return examples


def compute_bert_formatted_inputs(data_folder, max_seq_length=128):

    tokenizer, _, _, _ = load_bert(lower_case=True, base_version=True)

    # Generate formatted input
    for dataset_type in ['train', 'validation', 'test']:
        with open(Path(data_folder, f'spouse_{dataset_type}_set.json'), 'r') as f:
            dataset = json.load(f)

            # Just convert examples to InputExamples
            splitted_dataset = _process_examples(dataset)

            # Convert to BERT input
            examples_features = _convert_examples_to_features(
                examples=splitted_dataset, seq_length=max_seq_length, tokenizer=tokenizer)

            print(f'Len of parsed examples is {len(examples_features)}, just to double check')

            with open(Path(data_folder, f'spouse_formatted_{dataset_type}_dataset.pickle'), 'wb') as w:
                pickle.dump(examples_features, w)


def main():

    data_folder = '../../../data/Spouse'

    #train, validation, test = load_spouse(data_folder)

    #fix_documents(load_spacy(), data_folder)

    compute_bert_formatted_inputs(data_folder)


if __name__ == '__main__':
    main()