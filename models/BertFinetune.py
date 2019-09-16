# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import torch
import torch.nn as nn
from pytorch_pretrained_bert import BertTokenizer, BertModel


BERT_BASE_EMBEDDING_DIM = 768
BERT_LARGE_EMBEDDING_DIM = 1024

def load_bert(base_version=True, lower_case=True, device=None):

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


class BertFinetune(nn.Module):
    def __init__(self, input_size, num_classes):
        super(BertFinetune, self).__init__()
        self.linear = nn.Linear(input_size, num_classes)
        _, model, device, embedding_dim = load_bert()
        self.bert_model = model
        self.embedding_dim = embedding_dim

    def forward(self, *data):

        x, annotations, mask, batch_idx = data

        all_encoder_layers, _ = self.bert_model(x, attention_mask=mask, token_type_ids=None)

        last_layer = all_encoder_layers[-1]

        out = self.linear(last_layer[:, 0, :])

        return out, None
