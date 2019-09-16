# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import spacy
from spacy.pipeline.pipes import Sentencizer


def load_spacy():
    print('loading spacy...')
    spacy.prefer_gpu()
    pipeline = spacy.load('en_core_web_lg')
    sentencizer = Sentencizer()
    pipeline.add_pipe(sentencizer, first=True)
    print(pipeline.pipeline)  # list of the above

    return pipeline