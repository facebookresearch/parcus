# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import torch
import torch.nn as nn
from torch_scatter import scatter_add


class LogisticRegression(nn.Module):
    # For Hatespeech
    def __init__(self, input_size, num_classes):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_size, num_classes)

    def forward(self, *data):

        x, _ = data

        out = self.linear(x)
        return out, None


class LogisticRegressionOnTokens(LogisticRegression):

    def __init__(self, input_size, num_classes):
        super().__init__(input_size, num_classes)

    def forward(self, *data):

        x, annotations, mask, batch_idx = data

        src = self.linear(x)

        out = scatter_add(src, batch_idx.long(), dim=0)

        return out, src
