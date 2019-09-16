# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel

BERT_BASE_EMBEDDING_DIM = 768
BERT_LARGE_EMBEDDING_DIM = 1024


def load_bert(base_version=True, lower_case=True, device=None):

    if device is None:
        # If you have a GPU, put everything on cuda
        if torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'

    if base_version:
        embedding_dim = BERT_BASE_EMBEDDING_DIM
        if lower_case:
            bert_name = 'bert-base-uncased'
        else:
            bert_name = 'bert-base-cased'
    else:
        embedding_dim = BERT_LARGE_EMBEDDING_DIM
        if lower_case:
            bert_name = 'bert-large-uncased'
        else:
            bert_name = 'bert-large-cased'

    # Load pre-trained model tokenizer (vocabulary)
    tokenizer = BertTokenizer.from_pretrained(bert_name)
    # Load pre-trained model (weights)
    model = BertModel.from_pretrained(bert_name)

    return tokenizer, model, device, embedding_dim


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()
