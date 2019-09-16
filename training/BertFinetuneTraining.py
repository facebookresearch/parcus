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
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
import json
import numpy as np
from copy import deepcopy

from torch.utils.data.dataset import Subset, ConcatDataset

from datasets.BertFinetuneDataset import BertFinetuneDataset
from datasets.utils import custom_collate
from models.BertFinetune import BertFinetune
from training.utils import get_data_splits, _compute_all_targets, process_outputs


def train_model(train, model, optimizer, scheduler, max_epochs, loss_fun=torch.nn.CrossEntropyLoss(), device='cpu'):

    model.train()
    model.to(device)

    for epoch in range(1, max_epochs + 1):

        for inputs, annotations, mask_input, targets, _ in train:

            inputs, batch = inputs
            annotations, _ = annotations
            mask_input, _ = mask_input
            targets, _ = targets

            inputs = inputs.to(device)
            batch = batch.to(device)
            annotations = annotations.long().to(device)
            mask_input = mask_input.to(device)
            targets = targets.long().to(device)

            # Reset the gradient after a mini-batch update
            optimizer.zero_grad()

            # Run the forward pass.
            inputs = (inputs, annotations, mask_input, batch)
            out, _ = model(*inputs)

            # Compute the loss, gradients, and update the parameters by calling optimizer.step()
            loss = loss_fun(out, targets)

            loss.backward()
            optimizer.step()

            del inputs
            del batch
            del annotations
            del mask_input
            del targets
            del loss
            del out
            torch.cuda.empty_cache()

        if (epoch%1) == 0:
            print(f'Epoch {epoch}')

    return None


def predict(data, model, device='cpu'):

    outs = None
    all_targets = None

    model.to(device)
    model.eval()
    with torch.no_grad():

        for inputs, annotations, mask_input, targets, indices in data:

            inputs, batch = inputs
            annotations, _ = annotations
            mask_input, _ = mask_input
            targets, _ = targets

            inputs = inputs.to(device)
            batch = batch.to(device)
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

            del inputs
            del batch
            del annotations
            del mask_input
            del targets
            del out
            torch.cuda.empty_cache()

        ac, pr, rec, f1 = float(accuracy_score(all_targets, outs).item() * 100), float(precision_score(all_targets, outs).item() * 100), \
                            float(recall_score(all_targets, outs).item() * 100), float(f1_score(all_targets, outs).item() * 100)

        del all_targets
        del outs
        torch.cuda.empty_cache()

    return ac, pr, rec, f1



def compute(highlighted_train_set_path, non_highlighted_train_set_path, validation_set_path, test_set_path, no_runs, train_size, boostrap_split, train_split, validation_split, unlabelled_split,
            results_folder, score_to_optimize, dim_target=2):

    model_class = BertFinetune
    dataset_class = BertFinetuneDataset

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

    if torch.cuda.is_available():
        device = 'cuda:1'
        print('Using cuda')
    else:
        device = 'cpu'

    best_vl_scores = {'accuracy': 0,
                      'precision': 0,
                      'recall': 0,
                      'f1': 0}

    # These are our hyper-parameters
    best_params = None


    batch_sizes = [16] # Cannot use more than this ow memory error on Spouse

    model = None
    for batch_size in batch_sizes:
        for learning_rate in [2e-5, 3e-5, 5e-5]:
                for num_epochs in [10, 2, 4]:

                    vl_scores = {'accuracy': 0,
                                 'precision': 0,
                                 'recall': 0,
                                 'f1': 0}

                    for run in range(no_runs):
                        train_loader = DataLoader(train_set, batch_size=batch_size,
                                                  collate_fn=custom_collate, shuffle=True)
                        valid_loader = DataLoader(validation_set, batch_size=batch_size,
                                                  collate_fn=custom_collate, shuffle=False)
                        test_loader = DataLoader(test_set, batch_size=batch_size, collate_fn=custom_collate, shuffle=False)

                        model = model_class(input_size, dim_target)

                        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.)
                        # gamma = decaying factor
                        scheduler = StepLR(optimizer, step_size=50, gamma=1.)  # Useless scheduler

                        print('Start training')
                        train_model(train_loader, model, optimizer, scheduler,
                                                   max_epochs=num_epochs, device=device)

                        vl_acc, vl_pr, vl_rec, vl_f1 = predict(valid_loader, model, device)
                        print('End of prediction')


                        vl_scores['accuracy'] = vl_scores['accuracy'] + float(vl_acc)
                        vl_scores['precision'] = vl_scores['precision'] + float(vl_pr)
                        vl_scores['recall'] = vl_scores['recall'] + float(vl_rec)
                        vl_scores['f1'] = vl_scores['f1'] + float(vl_f1)

                        del model
                        del vl_acc
                        del vl_pr
                        del vl_rec
                        del vl_f1
                        del train_loader
                        del valid_loader
                        del test_loader
                        del scheduler
                        del optimizer
                        torch.cuda.empty_cache()

                    # AVERAGE OVER RUNS
                    for key in ['accuracy', 'precision', 'recall', 'f1']:
                        vl_scores[key] = vl_scores[key] / no_runs

                    if vl_scores[score_to_optimize] > best_vl_scores[score_to_optimize]:
                        print(f'Best on VL score: {vl_scores}')
                        best_vl_scores = deepcopy(vl_scores)
                        best_params = deepcopy(
                            {'learning_rate': learning_rate,
                             'train_split': train_split, 'valid_split': validation_split,
                             'batch_size': batch_size, 'epochs': num_epochs,
                             })
    te_scores = {
        'best_params': best_params,
        'best_vl_scores': best_vl_scores,
        'test_scores': {'accuracy': 0,
                        'precision': 0,
                        'recall': 0,
                        'f1': 0}
    }

    model = None
    for run in range(no_runs):
        if model is not None:
            del model
            torch.cuda.empty_cache()

        train_loader = DataLoader(train_set, batch_size=batch_size,
                                  collate_fn=custom_collate, shuffle=True)
        valid_loader = DataLoader(validation_set, batch_size=batch_size,
                                  collate_fn=custom_collate, shuffle=False)
        test_loader = DataLoader(test_set, batch_size=batch_size, collate_fn=custom_collate, shuffle=False)


        model = model_class(input_size, dim_target)

        optimizer = torch.optim.Adam(model.parameters(), lr=best_params['learning_rate'],
                                     weight_decay=0.)
        scheduler = StepLR(optimizer, step_size=50, gamma=1.)  # Useless scheduler for now

        epoch_losses = train_model(train_loader, model, optimizer, scheduler,
                                   max_epochs=best_params['epochs'], device=device)
        te_acc, te_pr, te_rec, te_f1 = predict(test_loader, model, device)

        te_scores['test_scores']['accuracy'] = te_scores['test_scores']['accuracy'] + float(te_acc)
        te_scores['test_scores']['precision'] = te_scores['test_scores']['precision'] + float(te_pr)
        te_scores['test_scores']['recall'] = te_scores['test_scores']['recall'] + float(te_rec)
        te_scores['test_scores']['f1'] = te_scores['test_scores']['f1'] + float(te_f1)

        del model
        del te_acc
        del te_pr
        del te_rec
        del te_f1
        del train_loader
        del valid_loader
        del test_loader
        del scheduler
        del optimizer
        torch.cuda.empty_cache()

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


def train(highlighted_set_path, non_highlighted_set_path, validation_set_path, test_set_path, train_sizes, dim_target,
          no_bootstraps, score_to_optimize, results_folder, seed=42, debug=False,
          max_workers=2):

    dataset_class = BertFinetuneDataset

    highlighted_set = dataset_class(highlighted_set_path)
    non_highlighted_set = dataset_class(non_highlighted_set_path)

    no_runs = 1

    # Generate same splits using seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    for train_size in train_sizes:

        pool = concurrent.futures.ProcessPoolExecutor(max_workers=max_workers)

        for boostrap_split in range(no_bootstraps):

            all_train_dataset, train_split, validation_split, unlabelled_split = \
                get_data_splits(highlighted_set, non_highlighted_set, train_size, target_idx=3)

            if not debug:
                pool.submit(compute, highlighted_set_path, non_highlighted_set_path, validation_set_path, test_set_path, int(no_runs), int(train_size),
                            int(boostrap_split),
                            list(train_split), list(validation_split), list(unlabelled_split), results_folder,
                            score_to_optimize, dim_target)
            else:  # DEBUG
                compute(highlighted_set_path, non_highlighted_set_path, validation_set_path, test_set_path, int(no_runs), int(train_size), int(boostrap_split),
                        list(train_split), list(validation_split), list(unlabelled_split), results_folder,
                        score_to_optimize, dim_target)

        pool.shutdown()  # wait the batch of configs to terminate

if __name__ == '__main__':
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    os.environ['OMP_NUM_THREADS'] = "1"

    # Hyper Parameters
    input_size = 768
    dim_target = 2
    no_bootstraps = 10
    score_to_optimize = 'f1'  # Use Accuracy for MovieReview
    seed = 42

    highlighted_train_dataset = str(Path(f'../data/Spouse/Spouse_FineTune/', f'Spouse_train/processed/highlighted'))
    non_highlighted_train_dataset = str(Path(f'../data/Spouse/Spouse_FineTune/', f'Spouse_train/processed/'))
    validation_dataset = str(Path(f'../data/Spouse/Spouse_FineTune/', f'Spouse_validation/processed/'))
    test_set = str(Path(f'../data/Spouse/Spouse_FineTune/', f'Spouse_test/processed/'))
    results_folder = 'results/Spouse_Finetune/BertFinetuning/'


    # Compute these first
    train_sizes = [10, 30, 60, 150, 300]
    train(highlighted_train_dataset, non_highlighted_train_dataset, validation_dataset,
          test_set, train_sizes, dim_target,
          no_bootstraps, score_to_optimize, Path(results_folder), seed,
          debug=True, max_workers=1)
