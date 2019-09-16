# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import concurrent.futures
import random
from pathlib import Path
import os
import torch
from metal import RandomSearchTuner
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
import json
import numpy as np
from copy import deepcopy

from torch.utils.data.dataset import Subset, ConcatDataset

from datasets.BertBaselineDataset import BertBaselineTokensDataset
from datasets.SpouseBaselineDataset import SpouseMLPDataset
from datasets.utils import custom_collate
from models.LabelModelNoSeed import LabelModelNoSeed
from models.LogisticRegression import LogisticRegression, LogisticRegressionOnTokens
from models.NPM import MLPOnTokens, MLPOnTokensWithHighlights, NBOW, DAN, NBOW2
from training.utils import get_data_splits, _compute_all_targets, process_outputs


def train_model(train, model, optimizer, scheduler, max_epochs, loss_fun=torch.nn.CrossEntropyLoss(), device='cpu'):

    model.train()
    epochs_loss = []
    for epoch in range(1, max_epochs + 1):

        epoch_losses = []
        accuracy = 0.
        tot_samples = 0.

        for inputs, annotations, mask_input, targets, _ in train:

            inputs, batch = inputs
            annotations, _ = annotations
            mask_input, _ = mask_input
            targets, _ = targets

            inputs = inputs.to(device)
            batch = batch.to(device)
            annotations = annotations.long().to(device)
            mask_input = mask_input.to(device)
            targets = targets.to(device)

            # Reset the gradient after a mini-batch update
            optimizer.zero_grad()

            # Run the forward pass.
            inputs = (inputs, annotations, mask_input, batch)
            out, _ = model(*inputs)

            # Compute the loss, gradients, and update the parameters by calling optimizer.step()
            loss = loss_fun(out, targets.long())

            accuracy += torch.sum(torch.argmax(out, dim=1).long() == targets.long()).float()
            tot_samples += targets.shape[0]

            loss.backward()
            optimizer.step()

            epoch_losses.append(float(loss))

            inputs = None
            batch = None
            annotations = None
            targets = None
            out = None
            loss = None

        epoch_avg_loss = sum(epoch_losses) / len(epoch_losses)
        epochs_loss.append(epoch_avg_loss)

    return epochs_loss


def predict(data, model, device='cpu'):

    outs = None
    all_targets = None

    model.eval()
    with torch.no_grad():

        for inputs, annotations, mask_input, targets, indices in data:

            inputs, batch = inputs
            annotations, _ = annotations
            mask_input, _ = mask_input
            targets, _ = targets

            inputs = inputs.to(device)
            batch = batch.to(device)
            annotations = annotations.long().to(device)
            mask_input = mask_input.to(device)
            targets = targets.to(device)

            # Run the forward pass.
            inputs = (inputs, None, mask_input, batch)
            out, _ = model(*inputs)

            if outs is None:
                outs = torch.argmax(out, dim=1).detach().cpu().numpy()
            else:
                outs = np.concatenate((outs, torch.argmax(out, dim=1).detach().cpu().numpy()), axis=0)

            if all_targets is None:
                all_targets = targets.detach().cpu().numpy()
            else:
                all_targets = np.concatenate((all_targets, targets.detach().cpu().numpy()), axis=0)

            inputs = None
            batch = None
            annotations = None
            targets = None
            out = None
            loss = None

    return accuracy_score(all_targets, outs) * 100, precision_score(all_targets, outs) * 100, \
           recall_score(all_targets, outs) * 100, f1_score(all_targets, outs) * 100, outs


def compute(model_string, dataset_string, highlighted_train_set_path, non_highlighted_train_set_path, validation_set_path, test_set_path, no_runs, train_size, boostrap_split, train_split, validation_split, unlabelled_split,
            results_folder, score_to_optimize, dim_target=2):

    if model_string == 'LogisticRegressionOnTokens':
        model_class = LogisticRegressionOnTokens
    elif model_string == 'MLPOnTokens':
        model_class = MLPOnTokens
    elif model_string == 'NBOW':
        model_class = NBOW
    elif model_string == 'NBOW2':
        model_class = NBOW2
    elif model_string == 'DAN':
        model_class = DAN
    elif model_string == 'MLPOnTokensWithHighlights':
        model_class = MLPOnTokensWithHighlights
    else:
        raise

    if dataset_string == 'BertBaselineTokensDataset':
        dataset_class = BertBaselineTokensDataset
    elif dataset_string == 'SpouseMLPDataset':
        dataset_class = SpouseMLPDataset
    else:
        raise

    highlighted_train_set = dataset_class(highlighted_train_set_path)
    non_highlighted_train_set = dataset_class(non_highlighted_train_set_path)
    test_set = dataset_class(test_set_path)

    all_train_dataset = ConcatDataset((highlighted_train_set, non_highlighted_train_set))

    train_set = Subset(all_train_dataset, train_split)

    if validation_set_path is not None:
        validation_set = dataset_class(validation_set_path)
        # Validation split is not interesting if we have an explicit validation set
        unlabelled_set = Subset(all_train_dataset, validation_split + unlabelled_split)
    else:
        validation_set = Subset(all_train_dataset, validation_split)
        unlabelled_set = Subset(all_train_dataset, unlabelled_split)

    test_loader = DataLoader(test_set, batch_size=batch_size, collate_fn=custom_collate, shuffle=False)

    best_vl_scores = {'accuracy': 0,
                      'precision': 0,
                      'recall': 0,
                      'f1': 0}

    # These are our hyper-parameters
    best_params = None

    if 'NBOW' in model_string:
        hidden_units = [0.]
        highlight_pow_bases = [None]
        tokens_dropout = [None]
    if model_string == 'DAN':
        hidden_units = [8, 32]
        highlight_pow_bases = [None]
        tokens_dropout = [0.5, 0.3, 0.]
    elif model_string == 'MLPOnTokens':
        hidden_units = [8, 32]
        highlight_pow_bases = [None]
        tokens_dropout = [None]
    elif model_string == 'MLPOnTokensWithHighlights':
        hidden_units = [8, 32]
        highlight_pow_bases = [float(np.exp(1)), 5, 10]
        tokens_dropout = [None]
    else:
        hidden_units = [None]
        tokens_dropout = [None]
        highlight_pow_bases = [None]

    for t_d in tokens_dropout:
        for hpb in highlight_pow_bases:
            for hidden in hidden_units:
                for learning_rate in [1e-2, 1e-3]:
                    for weight_decay in [1e-2, 1e-4]:
                        for num_epochs in [100, 500]:

                            vl_scores = {'accuracy': 0,
                                         'precision': 0,
                                         'recall': 0,
                                         'f1': 0}

                            for run in range(no_runs):
                                train_loader = DataLoader(train_set, batch_size=batch_size,
                                                          collate_fn=custom_collate, shuffle=True)
                                valid_loader = DataLoader(validation_set, batch_size=batch_size,
                                                          collate_fn=custom_collate, shuffle=False)

                                if hidden is not None:
                                    if hpb is not None:
                                        assert model_string == 'MLPOnTokensWithHighlights'
                                        model = model_class(input_size, dim_target, hidden, highlights_pow_base=hpb)  # For MLP ablation study
                                    elif t_d is None:
                                        assert model_string == 'MLPOnTokens' or model_string == 'NBOW'
                                        model = model_class(input_size, dim_target, hidden)
                                    elif t_d is not None:
                                        assert model_string == 'DAN'
                                        model = model_class(input_size, dim_target, hidden, t_d)

                                else:
                                    model = model_class(input_size, dim_target)

                                optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
                                # gamma = decaying factor
                                scheduler = StepLR(optimizer, step_size=50, gamma=1.)  # Useless scheduler

                                epoch_losses = train_model(train_loader, model, optimizer, scheduler,
                                                           max_epochs=num_epochs, device=device)

                                vl_acc, vl_pr, vl_rec, vl_f1, _ = predict(valid_loader, model, device)

                                vl_scores['accuracy'] = vl_scores['accuracy'] + float(vl_acc)
                                vl_scores['precision'] = vl_scores['precision'] + float(vl_pr)
                                vl_scores['recall'] = vl_scores['recall'] + float(vl_rec)
                                vl_scores['f1'] = vl_scores['f1'] + float(vl_f1)

                            # AVERAGE OVER RUNS
                            for key in ['accuracy', 'precision', 'recall', 'f1']:
                                vl_scores[key] = vl_scores[key] / no_runs

                            if vl_scores[score_to_optimize] > best_vl_scores[score_to_optimize]:
                                best_vl_scores = deepcopy(vl_scores)
                                best_params = deepcopy(
                                    {'learning_rate': learning_rate,
                                     'train_split': train_split, 'valid_split': validation_split,
                                     'weight_decay': weight_decay, 'epochs': num_epochs,
                                     'error_base': hpb,
                                     'tokens_dropout': t_d,
                                     'hidden_units': hidden})

    te_scores = {
        'best_params': best_params,
        'best_vl_scores': best_vl_scores,
        'test_scores': {'accuracy': 0,
                        'precision': 0,
                        'recall': 0,
                        'f1': 0}
    }

    for run in range(no_runs):
        if te_scores['best_params']['hidden_units'] is not None:
            if te_scores['best_params']['error_base'] is not None:
                assert model_string == 'MLPOnTokensWithHighlights'
                model = model_class(input_size, dim_target, te_scores['best_params']['hidden_units'], highlights_pow_base=te_scores['best_params']['error_base'])  # For MLP ablation study
            elif te_scores['best_params']['tokens_dropout'] is None:
                assert model_string == 'MLPOnTokens' or model_string == 'NBOW'
                model = model_class(input_size, dim_target, te_scores['best_params']['hidden_units'])  # For MLP ablation study
            elif te_scores['best_params']['tokens_dropout'] is not None:
                assert model_string == 'DAN'
                print('final on DAN')
                model = model_class(input_size, dim_target, te_scores['best_params']['hidden_units'], te_scores['best_params']['tokens_dropout'])
        else:
            model = model_class(input_size, dim_target)

        optimizer = torch.optim.Adam(model.parameters(), lr=best_params['learning_rate'],
                                     weight_decay=best_params['weight_decay'])
        scheduler = StepLR(optimizer, step_size=50, gamma=1.)  # Useless scheduler for now

        epoch_losses = train_model(train_loader, model, optimizer, scheduler,
                                   max_epochs=best_params['epochs'], device=device)
        te_acc, te_pr, te_rec, te_f1, _ = predict(test_loader, model, device)

        te_scores['test_scores']['accuracy'] = te_scores['test_scores']['accuracy'] + float(te_acc)
        te_scores['test_scores']['precision'] = te_scores['test_scores']['precision'] + float(te_pr)
        te_scores['test_scores']['recall'] = te_scores['test_scores']['recall'] + float(te_rec)
        te_scores['test_scores']['f1'] = te_scores['test_scores']['f1'] + float(te_f1)

    # AVERAGE OVER RUNS
    for key in ['accuracy', 'precision', 'recall', 'f1']:
        te_scores['test_scores'][key] = te_scores['test_scores'][key] / no_runs

    print(f'Best VL scores found is {best_vl_scores}')
    print(f'Best TE scores found is {te_scores["test_scores"]}')
    print(f'End of model assessment for train size {train_size}, test results are {te_scores}')

    if not os.path.exists(results_folder):
        os.makedirs(results_folder)

    with open(Path(results_folder, f'BertBaseline_size_{train_size}_runs_{no_runs}_test_results_bootstrap_{boostrap_split}.json'), 'w') as f:
        json.dump(te_scores, f)


def train(model_string, dataset_string, highlighted_set_path, non_highlighted_set_path, validation_set_path, test_set_path, train_sizes, dim_target,
          no_bootstraps, score_to_optimize, results_folder, seed=42, debug=False,
          max_workers=2):

    if dataset_string == 'BertBaselineTokensDataset':
        dataset_class = BertBaselineTokensDataset
    elif dataset_string == 'SpouseMLPDataset':
        dataset_class = SpouseMLPDataset
    else:
        raise

    highlighted_set = dataset_class(highlighted_set_path)
    non_highlighted_set = dataset_class(non_highlighted_set_path)

    no_runs = 3

    # Generate same splits using seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    pool = concurrent.futures.ProcessPoolExecutor(max_workers=max_workers)

    for train_size in train_sizes:

        for boostrap_split in range(no_bootstraps):

            all_train_dataset, train_split, validation_split, unlabelled_split = \
                get_data_splits(highlighted_set, non_highlighted_set, train_size, target_idx=3)

            if not debug:
                pool.submit(compute, model_string, dataset_string, highlighted_set_path, non_highlighted_set_path, validation_set_path, test_set_path, int(no_runs), int(train_size),
                            int(boostrap_split),
                            list(train_split), list(validation_split), list(unlabelled_split), results_folder,
                            score_to_optimize, dim_target)
            else:  # DEBUG
                compute(model_string, dataset_string, highlighted_set_path, non_highlighted_set_path, validation_set_path, test_set_path, int(no_runs), int(train_size), int(boostrap_split),
                        list(train_split), list(validation_split), list(unlabelled_split), results_folder,
                        score_to_optimize, dim_target)

    pool.shutdown()  # wait the batch of configs to terminate


if __name__ == '__main__':
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    os.environ['OMP_NUM_THREADS'] = "1"

    #if torch.cuda.is_available():
    #    device = 'cuda'
    #else:
    device = 'cpu'

    # Hyper Parameters
    input_size = 768
    dim_target = 2
    batch_size = 32
    no_bootstraps = 10
    score_to_optimize = 'f1'  # Use Accuracy for MovieReview
    seed = 42


    highlighted_train_dataset = str(Path(f'../data/Spouse/', f'Spouse_train/processed/highlighted'))
    non_highlighted_train_dataset = str(Path(f'../data/Spouse/', f'Spouse_train/processed/'))
    validation_dataset = str(Path(f'../data/Spouse/', f'Spouse_validation/processed/'))
    test_set = str(Path(f'../data/Spouse/', f'Spouse_test/processed/'))
    # Ablation with MLP with highlights

    dataset_string = 'SpouseMLPDataset'
    model_string = 'NBOW2'
    results_folder = 'results/Spouse/NBOW2/'


    # Compute these first
    train_sizes = [10, 30, 60, 150, 300]
    train(model_string, dataset_string, highlighted_train_dataset, non_highlighted_train_dataset, validation_dataset,
          test_set, train_sizes, dim_target,
          no_bootstraps, score_to_optimize, Path(results_folder), seed,
          debug=False, max_workers=30)


    # -------------------------------------------------------------------------------------------------------------- #

    highlighted_train_dataset = str(Path(f'../data/hatespeech/', f'hatespeech_train/processed/highlighted'))
    non_highlighted_train_dataset = str(Path(f'../data/hatespeech/', f'hatespeech_train/processed/'))
    validation_dataset = None
    test_set = str(Path(f'../data/hatespeech/', f'hatespeech_test/processed/'))
    # Ablation with MLP with highlights

    dataset_string = 'BertBaselineTokensDataset'
    model_string = 'NBOW2'
    results_folder = 'results/hatespeech/NBOW2/'

    # Compute these first
    train_sizes = [10, 20, 50, 100, 200]
    train(model_string, dataset_string, highlighted_train_dataset, non_highlighted_train_dataset, validation_dataset,
          test_set, train_sizes, dim_target,
          no_bootstraps, score_to_optimize, Path(results_folder), seed,
          debug=False, max_workers=30)

    # -------------------------------------------------------------------------------------------------------------- #

    highlighted_train_dataset = str(Path(f'../data/MovieReview/', f'moviereview_train/processed/'))
    non_highlighted_train_dataset = None
    validation_dataset = None
    test_set = str(Path(f'../data/MovieReview/', f'moviereview_test/processed/'))
    # Ablation with MLP with highlights


    score_to_optimize = 'accuracy'  # Use Accuracy for MovieReview

    dataset_string = 'BertBaselineTokensDataset'
    model_string = 'NBOW2'
    results_folder = 'results/MovieReview/NBOW2/'


    # Compute these first
    train_sizes = [10, 20, 50, 100, 200]
    train(model_string, dataset_string, highlighted_train_dataset, non_highlighted_train_dataset, validation_dataset,
          test_set, train_sizes, dim_target,
          no_bootstraps, score_to_optimize, Path(results_folder), seed,
          debug=False, max_workers=30)
