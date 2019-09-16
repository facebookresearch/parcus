# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import os
import json
import pickle
from pathlib import Path

from utils.bert import load_bert
from utils.spacy import load_spacy

# An example may be composed by multiple Input Features (i.e. sentences)
class MovieReviewInputFeatures(object):
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
class MovieReviewInputExample(object):

    def __init__(self, unique_id, target, text_a, text_b):
        self.unique_id = unique_id
        self.target = target
        self.text_a = text_a
        self.text_b = text_b


def load_moviereview(data_folder):
    no_rats_neg, no_rats_pos = 'noRats_neg', 'noRats_pos'
    rats_neg, rats_pos = 'withRats_neg', 'withRats_pos'

    # How to create the pairs of words. In order not to incur in computational issues, we split the document in sentences,
    # we create ordered pairs for each sentence and we combine all these pairs together.

    # TEST SET is made by the LAST 100 not annotated negatives and the LAST 100 not annotated positives

    annotated_dataset = []
    test_set = []  # should contain last 100 examples of all folders

    max_length = 0
    doc_lengths = []

    # Create training + validation set
    for polarity in [rats_neg, rats_pos]:
        for filename in sorted(os.listdir(Path(data_folder, 'review_polarity_rationales', polarity))):
            target = 1 if polarity == rats_pos else -1
            with open(Path(data_folder, 'review_polarity_rationales', polarity, filename), 'r') as f:
                data = f.read()
                annotated_dataset.append({'sample': data, 'target': target, 'filename': filename})

    # Create test set out of the entire data
    for polarity in [no_rats_neg, no_rats_pos]:
        polarity_set = []
        for filename in sorted(os.listdir(Path(data_folder, 'review_polarity_rationales', polarity))):
            target = 1 if polarity == no_rats_pos else -1
            with open(Path(data_folder, 'review_polarity_rationales', polarity, filename), 'r') as f:
                data = f.read()

                doc_len = len(data.split())  # Rough measure of length without tokenization. USE TOKENIZATION?
                doc_lengths.append(doc_len)
                max_length = max(max_length, doc_len)
                polarity_set.append({'sample': data, 'target': target, 'filename': filename})

        polarity_set = polarity_set[-100:]
        test_set.extend(polarity_set)

    return annotated_dataset, test_set


def split_sentences(pipeline, data_folder, train, test):

    for dataset_type, dataset in [('train', train), ('test', test)]:

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
            # print(new_example)

        print('')

        print(f'{len(dataset)} sentences have been splitted for {dataset_type} dataset')
        with open(Path(data_folder, f'splitted_{dataset_type}_sentences.json'), 'w', encoding='utf-8') as f:
            json.dump(dataset, f)


def _convert_examples_to_features(examples, seq_length, tokenizer):
    """Loads a data file into a list of `InputFeature`s."""

    # This variable holds a list of examples. Each example is a list of sentences in the form of "features"
    dataset_features = []
    example_no = 0
    for example in examples:
        # get example unique ID
        example_unique_id = example.unique_id

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

            # Account for [CLS] and [SEP] with "- 2"
            if len(parsed_example) > seq_length - 3:
                parsed_example = parsed_example[0:(seq_length - 3)]

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
            MovieReviewInputFeatures(
                unique_example_id=example_unique_id,
                unique_sentence_id=sentence_unique_id,
                tokens=tokens,
                annotations=annotations,
                input_ids=input_ids,
                input_mask=input_mask,
                input_type_ids=input_type_ids)
        )
        sentence_unique_id += 1

        dataset_features.append((example_target, example_features))

        example_no +=1
        print(f'Parsed example {example_no}')
    return dataset_features


def _process_examples(dataset):
    """Read a list of `InputExample`s from list."""
    examples = []
    unique_id = 0

    for doc in dataset:
        sample = doc['sample']
        target = doc['target']
        text_a = sample  # this is a list of sentences
        text_b = None  # we do not have pairs of sentences, we just need words embeddings for each document

        examples.append(
            MovieReviewInputExample(unique_id=unique_id, target=target, text_a=text_a, text_b=text_b))
        unique_id += 1
        # See convert_examples_to_features

    return examples


def compute_bert_formatted_inputs(data_folder, max_seq_length=512):
    tokenizer, _, _, _ = load_bert(lower_case=True, base_version=True)

    # Generate formatted input
    for dataset_type in ['train', 'test']:
        with open(Path(data_folder, f'splitted_{dataset_type}_sentences.json'), 'r', encoding='utf-8') as f:
            dataset = json.load(f)

            # Just convert examples to InputExamples
            splitted_dataset = _process_examples(dataset)

            # Convert to BERT input
            # examples_features is a list of list. The outer list holds examples (documents), the inner list holds one feature object per sentence
            examples_features = _convert_examples_to_features(
                examples=splitted_dataset, seq_length=max_seq_length, tokenizer=tokenizer)

            print(f'Len of parsed examples is {len(examples_features)}, just to double check')

            with open(Path(data_folder, f'formatted_{dataset_type}_finetune_dataset.pickle'), 'wb') as w:
                pickle.dump(examples_features, w)


if __name__ == '__main__':
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

    data_folder = '../../../data/MovieReview/'

    #compute_bert_formatted_inputs(data_folder)
