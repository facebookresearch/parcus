# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from metal import LabelModel


class LabelModelNoSeed(LabelModel):
    def __init__(self, k=2, **kwargs):
        super().__init__(k, **kwargs)

    # Overwrite this method, it causes troubles in our model selection procedure
    def _set_seed(self, seed):
        pass

