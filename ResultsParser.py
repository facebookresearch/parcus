# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import json
import os

import numpy as np
from pathlib import Path


def parse_results(results_folder):

    key_to_results = {}

    for filename in [f for f in sorted(os.listdir(results_folder)) if '.json' in f]:

        key = filename[:-6]
        #print(key)

        with open(Path(results_folder, filename), 'r') as f:
            if key not in key_to_results:
                key_to_results[key] = [json.load(f)]
            else:
                key_to_results[key].append(json.load(f))

    for key, test_results in key_to_results.items():

        acc = []
        pr = []
        rec = []
        f1 = []

        for test_scores in test_results:
            acc.append(test_scores['test_scores']['accuracy'])
            pr.append(test_scores['test_scores']['precision'])
            rec.append(test_scores['test_scores']['recall'])
            f1.append(test_scores['test_scores']['f1'])

        print(f'No of test results is {len(test_results)}')
        acc = np.array(acc)
        pr = np.array(pr)
        rec = np.array(rec)
        f1 = np.array(f1)

        print(f'Test results for {key} are:')
        print(f'Accuracy: {np.mean(acc)} ({np.std(acc)})')
        print(f'Precision: {np.mean(pr)} ({np.std(pr)})')
        print(f'Recall: {np.mean(rec)} ({np.std(rec)})')
        print(f'F1: {np.mean(f1)} ({np.std(f1)})')
        print('')

if __name__ == '__main__':

    results_folder = 'results/Spouse_Finetune/BertFinetuning/'
    print(results_folder)
    parse_results(results_folder)
