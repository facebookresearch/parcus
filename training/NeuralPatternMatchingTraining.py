# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import concurrent
import json
import os
import sys
import random
from copy import deepcopy

import torch
from metal import RandomSearchTuner
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score
from torch.utils.data import DataLoader
from pathlib import Path
import numpy as np
from torch.utils.data.dataset import Subset, ConcatDataset
from torch.optim.lr_scheduler import StepLR
from torch.nn import functional as F

from datasets.NREDataset import NREHatespeechDataset, NRESpouseDataset, NREMovieReviewDataset
from datasets.utils import SingleRunWeightedRandomSampler, custom_collate
from models.LabelModelNoSeed import LabelModelNoSeed
from models.NPM import NeuralPM, NeuralPMwoHighlights, NeuralPMSpouse, NeuralPMSpousewoHighlights, NeuralPMNoLogic, NeuralPMOnlyCosine, NeuralPMSpouseNoLogic, NeuralPMSpouseOnlyCosine
from training.utils import get_data_splits, process_outputs


def computeLossLF(data, model, reduction='none', device='cpu'):
    losses = []
    idx_order = []  # needed to keep track of the samples order (to update weights in boosting)
    outs = None
    all_targets = None

    loss_function = torch.nn.CrossEntropyLoss(reduction='none')

    model.eval()
    # print('Evaluation...')
    b_idx = 0
    with torch.no_grad():
        for inputs, annotations, mask_input, targets, indices in data:

            b_idx += 1
            #print(f'Processing batch {b_idx}...', end='')
            #print('\r', end='')

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
                outs = out.detach().cpu()
            else:
                outs = torch.cat((outs, out.detach().cpu()), dim=0)

            if all_targets is None:
                all_targets = targets.detach().cpu()
            else:
                all_targets = torch.cat((all_targets, targets.detach().cpu()), dim=0)

            # Compute the loss, gradients, and update the parameters by calling optimizer.step()
            loss = loss_function(out.detach(), targets.detach().long())

            loss = loss.detach().cpu().numpy()
            losses.extend(list(loss))
            idx_order.extend(list(indices))

            # This solves memory issues when on GPU
            inputs = None
            batch = None
            annotations = None
            targets = None
            loss = None

            #print('', end='')

        if reduction == 'mean':
            losses = sum(losses) / len(losses)

    return losses, idx_order, outs, all_targets


def trainLF(train,
            model, l1_coeff, optimizer, scheduler, max_epochs, validation=None, device='cpu'):

    loss_function = torch.nn.CrossEntropyLoss()

    model.train()
    print('Training...')

    epochs_loss = []
    for epoch in range(1, max_epochs + 1):  # again, normally you would NOT do 300 epochs, it is toy data

        epoch_losses = []

        b_idx = 0
        for inputs, annotations, mask_input, targets, _ in train:

            b_idx += 1

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
            loss = loss_function(out, targets.long())

            # L1 regularization on linear model to get sparse activations
            l1_norm = torch.norm(model.lin.weight, p=1)
            loss += l1_norm * l1_coeff

            loss.backward()
            optimizer.step()

            epoch_losses.append(float(loss))

            # This solves memory issues when on GPU
            inputs = None
            batch = None
            annotations = None
            targets = None
            out = None
            loss = None

        epoch_avg_loss = sum(epoch_losses) / len(epoch_losses)
        epochs_loss.append(epoch_avg_loss)

        if scheduler is not None:
            scheduler.step()

        if validation is not None and epoch % 10 == 0:
            valid_loss, _, _, _ = computeLossLF(validation, model, reduction='mean')  # default reduction is 'none'
            print(f'Epoch {epoch}, train avg loss is {epoch_avg_loss}, valid avg loss is {valid_loss}')
        elif epoch == 1 or epoch % 10 == 0:
            print(f'Epoch {epoch}, train avg loss is {epoch_avg_loss}')

    return epochs_loss


def compute_predictions_for_DP(models, data_loader, save_path, device='cpu'):
    predictions_mat = None  # Will have shape (# sentences, # LFs)

    for model in models:

        model.to(device)
        model.gating_param = model.gating_param.to(device)
        model.a_or = model.a_or.to(device)
        model.two = model.two.to(device)

        loss, _, outs, _ = computeLossLF(data_loader, model, reduction='mean')
        # Append outputs to outputs of each model
        # loss = loss.detach().cpu()
        preds = outs.detach().cpu()
        if predictions_mat is None:
            predictions_mat = preds
        else:
            predictions_mat = torch.cat((predictions_mat, preds), dim=0)
        try:
            if save_path is not None:

                save_path = Path(save_path)

                if not os.path.exists(save_path.parent):
                    os.makedirs(save_path.parent)

                torch.save(predictions_mat, save_path)
        except Exception as e:
            print(e)

    return predictions_mat, loss


def bagging(model_string, M, train_set, lr=1e-4, l1_coeff=1e-2, l2=1e-1, max_epochs=50, no_prototypes=3, gating_param=100,
            batch_size=32, embedding_dim=768, dim_target=2, highlights_pow_base=np.exp(1), save_path=None, device='cpu'):

    if model_string == 'NeuralPM':
        model_class = NeuralPM
    elif model_string == 'NeuralPMSpouse':
        model_class = NeuralPMSpouse
    elif model_string == 'NeuralPMwoHighlights':  # Ablation # 1
        model_class = NeuralPMwoHighlights
    elif model_string == 'NeuralPMSpousewoHighlights':  # Ablation # 1
        model_class = NeuralPMSpousewoHighlights
    elif model_string == 'NeuralPMNoLogic':
        model_class = NeuralPMNoLogic
    elif model_string == 'NeuralPMOnlyCosine':
        model_class = NeuralPMOnlyCosine
    elif model_string == 'NeuralPMSpouseNoLogic':
        model_class = NeuralPMSpouseNoLogic
    elif model_string == 'NeuralPMSpouseOnlyCosine':
        model_class = NeuralPMSpouseOnlyCosine
    else:
        raise

    # Determine the number of samples N
    N = len(train_set)

    models = []

    # Initialize weights
    weights = torch.ones(N)

    for i in range(M):

        # Normalized probabilities of weights
        p = weights / torch.sum(weights)
        # Give a bit of chance to other samples
        p[p == 0] = 1 / N

        if i > 1:
            # Define the sampler for this iteration
            sampler = SingleRunWeightedRandomSampler(weights=p, num_samples=N, replacement=True)

            # Build the data loader
            train_loader = DataLoader(train_set, batch_size=batch_size, collate_fn=custom_collate, sampler=sampler,
                                      pin_memory=False, num_workers=0)
        else:
            # The first LF will have access to all examples
            train_loader = DataLoader(train_set, batch_size=batch_size, collate_fn=custom_collate, shuffle=True,
                                      pin_memory=False, num_workers=0)

        # Istantiate the LF
        model = model_class(embedding_dim, dim_target, num_prototypes=no_prototypes, gating_param=gating_param,
                            highlights_pow_base=highlights_pow_base)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2)

        # gamma = decaying factor
        scheduler = StepLR(optimizer, step_size=100, gamma=0.9)

        # Move everything to device for computation
        model.to(device)
        model.gating_param = model.gating_param.to(device)
        model.a_or = model.a_or.to(device)
        model.two = model.two.to(device)

        # Train the new model
        epochs = trainLF(train_loader, model, l1_coeff, optimizer, scheduler, max_epochs)

        # Append new model to models list
        models.append(model)

    return models


def compute(model_string, dataset_string, highlighted_set_path, non_highlighted_set_path, validation_set_path, test_set_path, no_Lfs, train_size, boostrap_split, train_split, validation_split, unlabelled_split,
            results_folder, score_to_optimize, dim_target=2, rationale_noise=0):

    if dataset_string == 'NREHatespeechDataset':
        dataset_class = NREHatespeechDataset
    elif dataset_string == 'NRESpouseDataset':
        dataset_class = NRESpouseDataset
    elif dataset_string == 'NREMovieReviewDataset':
        dataset_class = NREMovieReviewDataset
    else:
        raise

    highlighted_set = dataset_class(highlighted_set_path, rationale_noise=rationale_noise)
    non_highlighted_set = dataset_class(non_highlighted_set_path, rationale_noise=rationale_noise)

    test_set = dataset_class(test_set_path)

    all_train_dataset = ConcatDataset((highlighted_set, non_highlighted_set))

    train_set = Subset(all_train_dataset, train_split)

    if validation_set_path is not None:
        validation_set = dataset_class(validation_set_path)
        # Validation split is not interesting if we have an explicit validation set
        unlabelled_set = Subset(all_train_dataset, validation_split + unlabelled_split)
    else:
        validation_set = Subset(all_train_dataset, validation_split)
        unlabelled_set = Subset(all_train_dataset, unlabelled_split)

    best_vl_scores = {'accuracy': 0.,
                      'precision': 0.,
                      'recall': 0.,
                      'f1': 0.}

    te_scores = {
        'best_params': {},
        'best_vl_scores': {},
        'test_scores': {'accuracy': 0,
                        'precision': 0,
                        'recall': 0,
                        'f1': 0}
    }

    if 'fasttext' in highlighted_set_path:
        embedding_dim = 300
    else:
        embedding_dim = 768

    for hpb in [float(np.exp(1)), 5, 10]:
        for lr in [1e-2]:
            for l1 in [1e-2, 1e-3]:
                for l2 in [1e-3, 1e-4]:
                    for no_prototypes in [5, 10]:
                        for num_epochs in [500]:
                            for gating_param in [10, 100]:


                                models = bagging(model_string, no_Lfs, train_set, \
                                                 lr=lr, l1_coeff=l1, l2=l2, max_epochs=num_epochs, \
                                                 no_prototypes=no_prototypes, embedding_dim=embedding_dim,
                                                 gating_param=gating_param, batch_size=32,
                                                 save_path=None, highlights_pow_base=hpb)

                                # Compute prediction for each NRE
                                for dataset_type, dataset in [('train', train_set), ('validation', validation_set),
                                                              ('test', test_set), ('unlabelled', unlabelled_set)]:

                                    if dataset_type == 'unlabelled' and no_Lfs == 1:
                                        continue

                                    # IMPORTANT: SHUFFLE MUST STAY FALSE (SEE TARGETS LATER)
                                    loader = DataLoader(dataset, batch_size=256, collate_fn=custom_collate, shuffle=False,
                                                        num_workers=0)
                                    predictions, _ = compute_predictions_for_DP(models, loader,
                                                                                save_path=Path(results_folder,
                                                                                               'stored_results',
                                                                                               f'predictions_{train_size}_size_{dataset_type}_{no_Lfs}_rules_bootstrap_{boostrap_split}.torch'))

                                # Compute and store targets on different files
                                for dataset, dataset_type in [(train_set, 'train'), (validation_set, 'validation'),
                                                              (test_set, 'test')]:

                                    dataset_loader = DataLoader(dataset, batch_size=256, collate_fn=custom_collate,
                                                                shuffle=False,
                                                                num_workers=2)

                                    all_targets = None
                                    for _, _, _, targets, _ in dataset_loader:
                                        targets, _ = targets

                                        if all_targets is None:
                                            all_targets = targets
                                        else:
                                            all_targets = torch.cat((all_targets, targets), dim=0)

                                    if not os.path.exists(Path(results_folder, 'stored_results')):
                                        os.makedirs(Path(results_folder, 'stored_results'))

                                    torch.save(all_targets,
                                               Path(results_folder, 'stored_results', f'all_targets_{dataset_type}_{train_size}_size_{no_Lfs}_rules_bootstrap_{boostrap_split}.torch'))

                                all_targets_valid = torch.load(
                                    Path(results_folder, 'stored_results', f'all_targets_validation_{train_size}_size_{no_Lfs}_rules_bootstrap_{boostrap_split}.torch'))
                                all_targets_test = torch.load(
                                    Path(results_folder, 'stored_results', f'all_targets_test_{train_size}_size_{no_Lfs}_rules_bootstrap_{boostrap_split}.torch'))

                                # For Data Programming
                                all_targets_valid_score = np.copy(all_targets_valid)  # It will be used to compute scores
                                all_targets_valid[all_targets_valid == 0] = 2
                                targets_valid = all_targets_valid.numpy()

                                all_targets_test_score = np.copy(all_targets_test)  # It will be used to compute scores
                                all_targets_test[all_targets_test == 0] = 2

                                train_predictions = torch.load(Path(results_folder, 'stored_results',
                                                                    f'predictions_{train_size}_size_train_{no_Lfs}_rules_bootstrap_{boostrap_split}.torch'),
                                                               map_location='cpu')

                                if no_Lfs != 1:
                                    unlabelled_predictions = torch.load(Path(results_folder, 'stored_results',
                                                                         f'predictions_{train_size}_size_unlabelled_{no_Lfs}_rules_bootstrap_{boostrap_split}.torch'),
                                                                    map_location='cpu')

                                valid_predictions = torch.load(Path(results_folder, 'stored_results',
                                                                    f'predictions_{train_size}_size_validation_{no_Lfs}_rules_bootstrap_{boostrap_split}.torch'),
                                                               map_location='cpu')

                                test_predictions = torch.load(Path(results_folder, 'stored_results',
                                                                   f'predictions_{train_size}_size_test_{no_Lfs}_rules_bootstrap_{boostrap_split}.torch'),
                                                              map_location='cpu')

                                train_predictions = train_predictions.cpu().reshape(-1, dim_target)

                                if no_Lfs != 1:
                                    unlabelled_predictions = unlabelled_predictions.cpu().reshape(-1, dim_target)

                                valid_predictions = valid_predictions.cpu().reshape(-1, dim_target)
                                test_predictions = test_predictions.cpu().reshape(-1, dim_target)

                                for threshold in [0.01, 0.05]:

                                    Ls_train = process_outputs(train_predictions, no_Lfs, threshold)
                                    if no_Lfs != 1:
                                        Ls_unlabelled = process_outputs(unlabelled_predictions, no_Lfs, threshold)
                                    Ls_valid = process_outputs(valid_predictions, no_Lfs, threshold)
                                    Ls_test = process_outputs(test_predictions, no_Lfs, threshold)

                                    if no_Lfs != 1:
                                        # Concatenate train and "unlabelled" data
                                        Ls_dataset = np.concatenate((Ls_train, Ls_unlabelled), axis=0)

                                        search_space = {
                                            'n_epochs': [100, 500],
                                            'lr': {'range': [0.01, 0.001], 'scale': 'log'},
                                            'show_plots': True,
                                        }

                                        tuner = RandomSearchTuner(LabelModelNoSeed)  # , seed=123)

                                        # ------------ DANGER ZONE: be careful here! ------------ #

                                        # Train on train+unlabelled because it is unsupervised (exploit unlabelled data), and "optimize" on
                                        # small validation set

                                        label_aggregator = tuner.search(
                                            search_space,
                                            train_args=[Ls_dataset],
                                            X_dev=Ls_valid, Y_dev=targets_valid.squeeze(),
                                            max_search=10, verbose=False, metric=score_to_optimize,
                                            shuffle=False
                                            # Leave it False, ow gen_splits generates different splits compared to linear baseline
                                        )

                                        Y_vl = label_aggregator.predict(Ls_valid)
                                        Y_test = label_aggregator.predict(Ls_test)

                                    # ------------ END OF DANGER ZONE ------------ #

                                    else:
                                        Y_vl = Ls_valid[:, 0]
                                        Y_test = Ls_test[:, 0]

                                    Y_vl[Y_vl == 2] = 0
                                    Y_test[Y_test == 2] = 0

                                    vl_pr = precision_score(all_targets_valid_score, Y_vl) * 100
                                    vl_rec = recall_score(all_targets_valid_score, Y_vl) * 100
                                    vl_acc = accuracy_score(all_targets_valid_score, Y_vl) * 100
                                    vl_f1 = f1_score(all_targets_valid_score, Y_vl) * 100

                                    te_pr = precision_score(all_targets_test_score, Y_test) * 100
                                    te_rec = recall_score(all_targets_test_score, Y_test) * 100
                                    te_acc = accuracy_score(all_targets_test_score, Y_test) * 100
                                    te_f1 = f1_score(all_targets_test_score, Y_test) * 100

                                    vl_scores = {'accuracy': float(vl_acc),
                                                 'precision': float(vl_pr),
                                                 'recall': float(vl_rec),
                                                 'f1': float(vl_f1)
                                                 }

                                    if vl_scores[score_to_optimize] > best_vl_scores[score_to_optimize]:
                                        best_vl_scores = deepcopy(vl_scores)
                                        best_params = deepcopy(
                                            {'learning_rate': lr, 'l1': l1, 'l2': l2,
                                             'train_split': train_split,
                                             'validation_split': validation_split,
                                             'no_prototypes': no_prototypes,
                                             'gating_param': gating_param,
                                             'threshold': threshold,
                                             'error_multiplier': hpb,
                                             'epochs': num_epochs})

                                        te_scores['best_params'] = best_params
                                        te_scores['best_vl_scores'] = best_vl_scores
                                        te_scores['test_scores'] = {'accuracy': float(te_acc),
                                                                    'precision': float(te_pr),
                                                                    'recall': float(te_rec),
                                                                    'f1': float(te_f1)}

    print(f'Best VL scores found is {best_vl_scores}')
    print(f'Best TE scores found is {te_scores["test_scores"]}')
    print(f'End of model assessment for train size {train_size} and {no_Lfs} rules, test results are {te_scores}')

    if not os.path.exists(results_folder):
        os.makedirs(results_folder)

    with open(
            Path(results_folder, f'NRE_size_{train_size}_rules_{no_Lfs}_test_results_bootstrap_{boostrap_split}.json'), 'w') as f:
        json.dump(te_scores, f)


def train(model_string, dataset_string, highlighted_set_path, non_highlighted_set_path, validation_set_path, test_set_path,
          train_sizes, LFs, results_folder, score_to_optimize, dim_target=2, no_bootstraps=10, seed=42, debug=False,
          max_workers=2, rationale_noise=0.):

    if dataset_string == 'NREHatespeechDataset':
        dataset_class = NREHatespeechDataset
    elif dataset_string == 'NRESpouseDataset':
        dataset_class = NRESpouseDataset
    elif dataset_string == 'NREMovieReviewDataset':
        dataset_class = NREMovieReviewDataset
    else:
        raise

    highlighted_set = dataset_class(highlighted_set_path)
    non_highlighted_set = dataset_class(non_highlighted_set_path)

    if not os.path.exists(results_folder):
        os.makedirs(results_folder)

    for no_Lfs in LFs:

        # Generate same splits using seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        pool = concurrent.futures.ProcessPoolExecutor(max_workers=max_workers)

        for train_size in train_sizes:  # Train_size must be less than len(train_idxs)

            for boostrap_split in range(no_bootstraps):

                all_train_dataset, train_split, validation_split, unlabelled_split = \
                    get_data_splits(highlighted_set, non_highlighted_set, train_size, target_idx=3)

                if not debug:
                    pool.submit(compute, model_string, dataset_string, highlighted_set_path, non_highlighted_set_path, validation_set_path, test_set_path, int(no_Lfs), int(train_size), int(boostrap_split),
                            list(train_split), list(validation_split), list(unlabelled_split), results_folder,
                            score_to_optimize, dim_target, rationale_noise)
                else:  # DEBUG
                    compute(model_string, dataset_string, highlighted_set_path, non_highlighted_set_path, validation_set_path, test_set_path, int(no_Lfs), int(train_size), int(boostrap_split),
                            list(train_split), list(validation_split), list(unlabelled_split), results_folder,
                            score_to_optimize, dim_target, rationale_noise)

        pool.shutdown()  # wait the batch of configs to terminate

if __name__ == '__main__':
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    os.environ['OMP_NUM_THREADS'] = "1"
    seed = 42

    '''
    score_to_optimize = 'f1'
    highlighted_train_dataset = str(Path(f'../data/Spouse/', f'Spouse_train/processed/highlighted'))
    non_highlighted_train_dataset = str(Path(f'../data/Spouse/', f'Spouse_train/processed/'))
    validation_dataset = str(Path(f'../data/Spouse/', f'Spouse_validation/processed/'))
    test_dataset =str(Path(f'../data/Spouse/', f'Spouse_test/processed/'))
    dataset_string = 'NRESpouseDataset'

    for model_string in ['NeuralPMSpousewoHighlights']:
        results_folder = f'results/Spouse/AdaptiveLFs/{model_string}'

        train_sizes = [10, 30, 60, 150, 300]
        LFs = [1]

        train(model_string, dataset_string, highlighted_train_dataset, non_highlighted_train_dataset, validation_dataset, test_dataset,
               train_sizes, LFs, results_folder=Path(results_folder), score_to_optimize=score_to_optimize,
               no_bootstraps=10, seed=seed, max_workers=30, debug=True, rationale_noise=0.)
    '''
    score_to_optimize = 'f1'
    highlighted_train_dataset = str(Path(f'../../data/hatespeech/', f'hatespeech_train/processed/highlighted'))
    non_highlighted_train_dataset = str(Path(f'../../data/hatespeech/', f'hatespeech_train/processed/'))
    validation_dataset = None
    test_dataset =str(Path(f'../../data/hatespeech/', f'hatespeech_test/processed/'))
    dataset_string = 'NREHatespeechDataset'

    for model_string in ['NeuralPMOnlyCosine']:
        results_folder = f'results/hatespeech/AdaptiveLFs/{model_string}'

        train_sizes = [10, 20, 50, 100, 200]
        LFs = [1]

        train(model_string, dataset_string, highlighted_train_dataset, non_highlighted_train_dataset, validation_dataset, test_dataset,
               train_sizes, LFs, results_folder=Path(results_folder), score_to_optimize=score_to_optimize,
               no_bootstraps=10, seed=seed, max_workers=30, debug=True, rationale_noise=0.)

    '''
    score_to_optimize = 'f1'
    highlighted_train_dataset = str(Path(f'../../data/hatespeech/', f'hatespeech_train_fasttext/processed/highlighted'))
    non_highlighted_train_dataset = str(Path(f'../../data/hatespeech/', f'hatespeech_train_fasttext/processed/'))
    validation_dataset = None
    test_dataset =str(Path(f'../../data/hatespeech/', f'hatespeech_test_fasttext/processed/'))
    dataset_string = 'NREHatespeechDataset'

    for model_string in ['NeuralPM']:
        results_folder = f'results/hatespeech/AdaptiveLFs/{model_string}Fasttext'

        train_sizes = [10, 20, 50, 100, 200]
        LFs = [1]

        train(model_string, dataset_string, highlighted_train_dataset, non_highlighted_train_dataset, validation_dataset, test_dataset,
              train_sizes, LFs, results_folder=Path(results_folder), score_to_optimize=score_to_optimize,
               no_bootstraps=10, seed=seed, max_workers=30, debug=False, rationale_noise=0.)
    '''