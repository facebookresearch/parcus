# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import torch
from torch_scatter import scatter_add, scatter_mean
from torch.autograd import Function
import torch.nn.functional as F

from models.LogisticRegression import LogisticRegression


class WeightedIdentity(Function):
    """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
    """

    @staticmethod
    def compute_weight_vector(annotations, base):
        # avoid having 0 when no tokens are highlighted
        if base is None:
            return torch.exp(torch.sum(annotations.float(), dim=1))
        else:
            return torch.pow(base, torch.sum(annotations.float(), dim=1))

    @staticmethod
    def forward(ctx, input, annotations, base=None):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        ctx.save_for_backward(annotations, base)
        return torch.mul(input, torch.pow(base, annotations.float()))

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        annotations, base = ctx.saved_tensors

        grad_input = grad_output.clone()

        # Weight each pairs according to the correspondent annotation
        sample_weight = WeightedIdentity.compute_weight_vector(annotations, base)
        grad_input = torch.mul(grad_input, sample_weight.unsqueeze(dim=1))

        # forward took 2 arguments, backward is expecting 2 backward arguments.
        # Just return None for annotations and base
        return grad_input, None, None


class NeuralPM(torch.nn.Module):
    """
    Labeling Function that takes 2 words (to be parametrized) and returns a score.
    We assume independence between prototypes, that is at first this holds because they are random,
    but then this assumption allows us to easily model "probability of a word of being similar
    to a prototype"  and interpret the model.
    """

    def __init__(self, dim_features, dim_targets, gating_param=100, highlights_pow_base=None, num_prototypes=3):
        """
        LF
        :param dim_features:
        :param a: default:2
        """
        super().__init__()
        self.dim_features = dim_features
        self.dim_targets = dim_targets

        if highlights_pow_base is not None:
            self.highlights_pow_base = torch.nn.Parameter(torch.Tensor([highlights_pow_base]))
            self.highlights_pow_base.requires_grad = False
        else:
            self.highlights_pow_base = None

        self.gating_param = torch.nn.Parameter(torch.Tensor([gating_param]))
        self.gating_param.requires_grad = False
        self.a_or = torch.nn.Parameter(torch.Tensor([2]))
        self.a_or.requires_grad = False
        self.two = torch.nn.Parameter(torch.Tensor([2]))
        self.two.requires_grad = False

        # learnable prototypes for token embedding
        self.prototypes = torch.nn.Parameter(torch.rand(num_prototypes, dim_features))
        self.prototypes.requires_grad = True

        self.no_logic_feats = num_prototypes * 2 + 2

        # Bias is very important here
        self.lin = torch.nn.Linear(self.no_logic_feats, dim_targets, bias=False)

    def _compute_delta(self, w, p):
        """
        Fires when word has similar meaning with respect to prototype
        :param w: the word/concept vector representation
        :param p: a prototype
        """
        # w has size (#pairs, dim_embedding), p has size (num_prototypes, dim-embedding)
        w_exp = w.unsqueeze(1)
        p_exp = p.unsqueeze(0)
        return self._activation(F.cosine_similarity(w_exp, p_exp, dim=2))

    def _compute_not_delta(self, w, p):
        """
        Fires when word has opposite meaning with respect to prototype
        :param w: the word/concept vector representation
        :param p: a prototype
        """
        # w has size (#pairs, dim_embedding), p has size (num_prototypes, dim-embedding)
        w_exp = w.unsqueeze(1)
        p_exp = p.unsqueeze(0)

        return self._activation(-F.cosine_similarity(w_exp, p_exp, dim=2))

    def _or_auxiliary(self, d):
        # size (#inputs, num_prototypes)
        return torch.pow((d - 2), -2 * self.a_or) - torch.pow(self.two, -2 * self.a_or) * (1 - d)

    def _or(self, deltas, and_f):
        # both have size (?, num_prototypes)
        tmp1 = self._or_auxiliary(deltas)

        return torch.sum(tmp1, dim=1) - and_f

    def _xor(self, and_f, or_f):
        # both have size (#pairs, 1)
        return or_f - and_f

    def _activation(self, out):
        return torch.pow(self.gating_param, out - 1)

    def forward(self, *data):

        x, annotations, mask, batch_idx = data

        deltas = self._compute_delta(x, self.prototypes)
        not_deltas = self._compute_not_delta(x, self.prototypes)

        # and_deltas = torch.prod(deltas, dim=1)
        # or_deltas = self._or(deltas, and_deltas)
        and_deltas, _ = torch.min(deltas, dim=1)
        or_deltas, _ = torch.max(deltas, dim=1)
        xor_deltas = self._xor(and_deltas, or_deltas)

        all_feats = torch.cat((
            deltas, not_deltas,
            and_deltas.unsqueeze(1),
            or_deltas.unsqueeze(1),

            # more_feats[:, 2:],
        ), dim=1)

        src = self.lin(all_feats).squeeze()

        # Apply importance weighting to word pairs, then tanh
        if annotations is not None:
            src = WeightedIdentity.apply(src, annotations, self.highlights_pow_base)

        out = scatter_add(src, batch_idx.long(), dim=0)
        return out, src


class NeuralPMSpouse(NeuralPM):

    def __init__(self, dim_features, dim_targets, gating_param=100, highlights_pow_base=None, num_prototypes=3):
        """
        LF
        :param dim_features:
        :param a: default:2
        """
        super().__init__(dim_features, dim_targets, gating_param, highlights_pow_base, num_prototypes)

    def forward(self, *data):

        x, annotations, mask, batch_idx = data

        deltas = self._compute_delta(x, self.prototypes)
        not_deltas = self._compute_not_delta(x, self.prototypes)

        # and_deltas = torch.prod(deltas, dim=1)
        # or_deltas = self._or(deltas, and_deltas)
        and_deltas, _ = torch.min(deltas, dim=1)
        or_deltas, _ = torch.max(deltas, dim=1)
        xor_deltas = self._xor(and_deltas, or_deltas)

        all_feats = torch.cat((
            deltas, not_deltas,
            and_deltas.unsqueeze(1),
            or_deltas.unsqueeze(1),

            # more_feats[:, 2:],
        ), dim=1)

        src = self.lin(all_feats).squeeze()

        # Apply importance weighting to word pairs
        if annotations is not None:
            src = WeightedIdentity.apply(src, annotations, self.highlights_pow_base)

        # Apply masking at INFERENCE time only! Allow training to see more tokens

        if not self.training:
            more_than_one = torch.sum(mask.float(), dim=1, keepdim=True)
            more_than_one = more_than_one >= 1
            # Apply masking also here.
            src = torch.mul(src, more_than_one.float())

        out = scatter_add(src, batch_idx.long(), dim=0)
        return out, src


# TODO will need to refactor at some point
class NeuralPMwoHighlights(NeuralPM):
    def __init__(self, dim_features, dim_targets, gating_param=100, highlights_pow_base=None, num_prototypes=3):
        super().__init__(dim_features, dim_targets, gating_param, highlights_pow_base, num_prototypes)

    def forward(self, *data):
        x, annotations, mask, batch_idx = data

        # Zeroing the contribution of annotations
        if annotations is not None:
            new_input = (x, annotations*0, mask, batch_idx)
        else:
            new_input = (x, None, mask, batch_idx)

        return super().forward(*new_input)


class NeuralPMSpousewoHighlights(NeuralPMSpouse):
    def __init__(self, dim_features, dim_targets, gating_param=100, highlights_pow_base=None, num_prototypes=3):
        super().__init__(dim_features, dim_targets, gating_param, highlights_pow_base, num_prototypes)

    def forward(self, *data):
        x, annotations, mask, batch_idx = data

        if annotations is None:
            new_input = (x, annotations, mask, batch_idx)
        else:
            # Zeroing the contribution of annotations
            new_input = (x, annotations*0, mask, batch_idx)


        return super().forward(*new_input)


class NBOW(LogisticRegression):

    def __init__(self, input_size, num_classes, hidden_units=0):
        super().__init__(input_size, num_classes)

        self.lin = torch.nn.Linear(input_size, num_classes)

    def forward(self, *data):

        x, annotations, mask, batch_idx = data

        # Cannot apply mask on tokens (Spouse) before taking the average, because that could greatly affect the mean!
        src = scatter_mean(x, batch_idx.long(), dim=0)

        out = self.lin(src)

        return out, src


class NBOW2(LogisticRegression):

    def __init__(self, input_size, num_classes, hidden_units=0):
        super().__init__(input_size, num_classes)

        self.anchor = self.prototypes = torch.nn.Parameter(torch.rand(input_size, 1))
        self.anchor.requires_grad = True

        self.lin = torch.nn.Linear(input_size, num_classes)

    def forward(self, *data):

        x, annotations, mask, batch_idx = data

        a_w = torch.mm(x, self.anchor)
        x = torch.mul(x, a_w)

        # Cannot apply mask on tokens (Spouse) before taking the average, because that could greatly affect the mean!
        src = scatter_mean(x, batch_idx.long(), dim=0)

        out = self.lin(src)

        return out, src


class DAN(LogisticRegression):

    def __init__(self, input_size, num_classes, hidden_units=8, tokens_dropout=0.):
        super().__init__(input_size, num_classes)

        self.tokens_dropout = tokens_dropout

        # two layer DAN (less capacity should work better on small datasets)
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(input_size, hidden_units),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_units, hidden_units),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_units, num_classes),
        )

    def forward(self, *data):

        x, annotations, mask, batch_idx = data

        # print(x.shape, batch_idx.shape, self.tokens_dropout)

        # Drop percentage of tokens
        dropout_mask = torch.rand(x.shape[0]) >= self.tokens_dropout

        x = x[dropout_mask]
        batch_idx = batch_idx[dropout_mask]

        # print(x.shape, batch_idx.shape)

        # Cannot apply mask on tokens (Spouse) before taking the average, because that could greatly affect the mean!
        src = scatter_mean(x, batch_idx.long(), dim=0)

        out = self.layers(src)

        return out, src

class MLPOnTokens(LogisticRegression):

    def __init__(self, input_size, num_classes, hidden_units=16, highlights_pow_base=None, use_highlights=False):
        super().__init__(input_size, num_classes)
        self.highlights_pow_base = highlights_pow_base
        self.use_highlights = use_highlights

        self.layers = torch.nn.Sequential(
            torch.nn.Linear(input_size, hidden_units),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_units, num_classes)
        )

        if highlights_pow_base is not None:
            self.highlights_pow_base = torch.nn.Parameter(torch.Tensor([highlights_pow_base]))
            self.highlights_pow_base.requires_grad = False
        else:
            self.highlights_pow_base = None


    def forward(self, *data):

        x, annotations, mask, batch_idx = data

        src = self.layers(x)

        if self.use_highlights and annotations is not None:
            src = WeightedIdentity.apply(src, annotations, self.highlights_pow_base)

        # Apply masking at INFERENCE time only! Allow training to see more tokens
        # Used for Spouse
        #if not self.training:
        #    more_than_one = torch.sum(mask.float(), dim=1, keepdim=True)
        #    more_than_one = more_than_one >= 1
        #    # Apply masking also here.
        #    src = torch.mul(src, more_than_one.float())

        out = scatter_add(src, batch_idx.long(), dim=0)

        return out, src


class MLPOnTokensWithHighlights(MLPOnTokens):
    def __init__(self, input_size, num_classes, hidden_units=16, highlights_pow_base=None):
        super().__init__(input_size, num_classes, hidden_units, highlights_pow_base,
                         use_highlights=True)


class NeuralPMNoLogic(NeuralPM):
    """
    Labeling Function that takes 2 words (to be parametrized) and returns a score.
    We assume independence between prototypes, that is at first this holds because they are random,
    but then this assumption allows us to easily model "probability of a word of being similar
    to a prototype"  and interpret the model.
    """

    def __init__(self, dim_features, dim_targets, gating_param=100, highlights_pow_base=None, num_prototypes=3):
        """
        LF
        :param dim_features:
        :param a: default:2
        """
        super().__init__(dim_features, dim_targets, gating_param, highlights_pow_base, num_prototypes)
        self.no_logic_feats = num_prototypes * 2
        self.lin = torch.nn.Linear(self.no_logic_feats, dim_targets, bias=False)

    def forward(self, *data):

        x, annotations, mask, batch_idx = data

        deltas = self._compute_delta(x, self.prototypes)
        not_deltas = self._compute_not_delta(x, self.prototypes)

        all_feats = torch.cat((
            deltas, not_deltas,
            # more_feats[:, 2:],
        ), dim=1)

        src = self.lin(all_feats).squeeze()

        # Apply importance weighting to word pairs, then tanh
        if annotations is not None:
            src = WeightedIdentity.apply(src, annotations, self.highlights_pow_base)

        out = scatter_add(src, batch_idx.long(), dim=0)
        return out, src


class NeuralPMOnlyCosine(NeuralPM):
    """
    Labeling Function that takes 2 words (to be parametrized) and returns a score.
    We assume independence between prototypes, that is at first this holds because they are random,
    but then this assumption allows us to easily model "probability of a word of being similar
    to a prototype"  and interpret the model.
    """

    def __init__(self, dim_features, dim_targets, gating_param=100, highlights_pow_base=None, num_prototypes=3):
        """
        LF
        :param dim_features:
        :param a: default:2
        """
        super().__init__(dim_features, dim_targets, gating_param, highlights_pow_base, num_prototypes)
        self.no_logic_feats = num_prototypes
        self.lin = torch.nn.Linear(self.no_logic_feats, dim_targets, bias=False)

    def forward(self, *data):

        x, annotations, mask, batch_idx = data

        deltas = self._compute_delta(x, self.prototypes)

        all_feats = deltas

        src = self.lin(all_feats).squeeze()

        # Apply importance weighting to word pairs, then tanh
        if annotations is not None:
            src = WeightedIdentity.apply(src, annotations, self.highlights_pow_base)

        out = scatter_add(src, batch_idx.long(), dim=0)
        return out, src


class NeuralPMSpouseOnlyCosine(NeuralPM):

    def __init__(self, dim_features, dim_targets, gating_param=100, highlights_pow_base=None, num_prototypes=3):
        """
        LF
        :param dim_features:
        :param a: default:2
        """
        super().__init__(dim_features, dim_targets, gating_param, highlights_pow_base, num_prototypes)
        self.no_logic_feats = num_prototypes
        self.lin = torch.nn.Linear(self.no_logic_feats, dim_targets, bias=False)

    def forward(self, *data):

        x, annotations, mask, batch_idx = data

        deltas = self._compute_delta(x, self.prototypes)

        all_feats = torch.cat((
            deltas,

            # more_feats[:, 2:],
        ), dim=1)

        src = self.lin(all_feats).squeeze()

        # Apply importance weighting to word pairs
        if annotations is not None:
            src = WeightedIdentity.apply(src, annotations, self.highlights_pow_base)

        # Apply masking at INFERENCE time only! Allow training to see more tokens

        if not self.training:
            more_than_one = torch.sum(mask.float(), dim=1, keepdim=True)
            more_than_one = more_than_one >= 1
            # Apply masking also here.
            src = torch.mul(src, more_than_one.float())

        out = scatter_add(src, batch_idx.long(), dim=0)
        return out, src


class NeuralPMSpouseNoLogic(NeuralPM):
    """
    Labeling Function that takes 2 words (to be parametrized) and returns a score.
    We assume independence between prototypes, that is at first this holds because they are random,
    but then this assumption allows us to easily model "probability of a word of being similar
    to a prototype"  and interpret the model.
    """

    def __init__(self, dim_features, dim_targets, gating_param=100, highlights_pow_base=None, num_prototypes=3):
        """
        LF
        :param dim_features:
        :param a: default:2
        """
        super().__init__(dim_features, dim_targets, gating_param, highlights_pow_base, num_prototypes)
        self.no_logic_feats = num_prototypes * 2
        self.lin = torch.nn.Linear(self.no_logic_feats, dim_targets, bias=False)

    def forward(self, *data):

        x, annotations, mask, batch_idx = data

        deltas = self._compute_delta(x, self.prototypes)
        not_deltas = self._compute_not_delta(x, self.prototypes)

        all_feats = torch.cat((
            deltas, not_deltas,
            # more_feats[:, 2:],
        ), dim=1)

        src = self.lin(all_feats).squeeze()

        # Apply importance weighting to word pairs, then tanh
        if annotations is not None:
            src = WeightedIdentity.apply(src, annotations, self.highlights_pow_base)

        if not self.training:
            more_than_one = torch.sum(mask.float(), dim=1, keepdim=True)
            more_than_one = more_than_one >= 1
            # Apply masking also here.
            src = torch.mul(src, more_than_one.float())

        out = scatter_add(src, batch_idx.long(), dim=0)
        return out, src
