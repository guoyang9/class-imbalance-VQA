import torch
from torch import nn
from torch.nn import functional as F

import utils.config as config


def convert_sigmoid_logits_to_binary_logprobs(logits):
    """Computes log(sigmoid(logits)), log(1-sigmoid(logits))."""
    log_prob = -F.softplus(-logits)
    log_one_minus_prob = -logits + log_prob
    return log_prob, log_one_minus_prob


def cross_entropy_loss(logits, labels, **kwargs):
    """ Modified cross entropy loss. """
    nll = F.log_softmax(logits, dim=-1)
    loss = -nll * labels / 10
    if 'miu' in kwargs:
        loss = loss * smooth(kwargs['miu'], kwargs['mask'])
    return loss.sum(dim=-1).mean()


def elementwise_logsumexp(a, b):
    """computes log(exp(x) + exp(b))"""
    return torch.max(a, b) + torch.log1p(torch.exp(-torch.abs(a - b)))


def renormalize_binary_logits(a, b):
    """Normalize so exp(a) + exp(b) == 1"""
    norm = elementwise_logsumexp(a, b)
    return a - norm, b - norm


def smooth(miu, mask):
    miu_valid = miu * mask
    miu_invalid = miu * (1.0 - mask) # most 1.0
    return miu_invalid + torch.clamp(F.softplus(miu_valid), max=100.0)


class Plain(nn.Module):
    def forward(self, logits, labels, **kwargs):
        if config.loss_type == 'ce':
            loss = cross_entropy_loss(logits, labels, **kwargs)
        else:
            if 'miu' in kwargs:
                loss = F.binary_cross_entropy_with_logits(logits, labels,
                            pos_weight=smooth(kwargs['miu'], kwargs['mask']))
            else:
                loss = F.binary_cross_entropy_with_logits(logits, labels)
            loss *= labels.size(1)
        return loss


class LearnedMixin(nn.Module):
    def __init__(self, hid_size=1024, smooth=True, smooth_init=-1, constant_smooth=0.0):
        """
        w: Weight of the entropy penalty
        smooth: Add a learned sigmoid(a) factor to the bias to smooth it
        smooth_init: How to initialize `a`
        constant_smooth: Constant to add to the bias to smooth it
        """
        super(LearnedMixin, self).__init__()
        self.smooth_init = smooth_init
        self.constant_smooth = constant_smooth
        self.bias_lin = torch.nn.Linear(hid_size, 1)
        self.smooth = smooth
        if self.smooth:
            self.smooth_param = torch.nn.Parameter(
                smooth_init * torch.ones((1,), dtype=torch.float32))
        else:
            self.smooth_param = None

    def bias_convert(self, **kwargs):
        factor = self.bias_lin.forward(kwargs['hidden'])  # [batch, 1]
        factor = F.softplus(factor)

        bias = torch.stack([kwargs['bias'], 1 - kwargs['bias']], 2)  # [batch, n_answers, 2]

        # Smooth
        bias += self.constant_smooth
        if self.smooth:
            soften_factor = torch.sigmoid(self.smooth_param)
            bias = bias + soften_factor.unsqueeze(1)

        bias = torch.log(bias)  # Convert to logspace

        # Scale by the factor
        # [batch, n_answers, 2] * [batch, 1, 1] -> [batch, n_answers, 2]
        bias = bias * factor.unsqueeze(1)
        return bias

    def loss_compute(self, logits, labels, bias_converted, **kwargs):
        log_prob, log_one_minus_prob = convert_sigmoid_logits_to_binary_logprobs(logits)
        log_probs = torch.stack([log_prob, log_one_minus_prob], 2)

        # Add the bias in
        logits = bias_converted + log_probs

        # Renormalize to get log probabilities
        log_prob, log_one_minus_prob = renormalize_binary_logits(logits[:, :, 0], logits[:, :, 1])

        # Compute loss
        loss_single = -(log_prob * labels + (1 - labels) * log_one_minus_prob)
        if 'miu' in kwargs:
            loss_single = loss_single * smooth(kwargs['miu'], kwargs['mask'])
        loss = loss_single.sum(1).mean(0)
        return loss

    def forward(self, logits, labels, **kwargs):
        bias_converted = self.bias_convert(**kwargs)
        loss = self.loss_compute(logits, labels, bias_converted, **kwargs)
        return loss


class LearnedMixinH(LearnedMixin):
    def __init__(self, hid_size=1024, smooth=True, smooth_init=-1, constant_smooth=0.0, w=0.36):
        """
        w: Weight of the entropy penalty
        smooth: Add a learned sigmoid(a) factor to the bias to smooth it
        smooth_init: How to initialize `a`
        constant_smooth: Constant to add to the bias to smooth it
        """
        super(LearnedMixinH, self).__init__(hid_size, smooth, smooth_init, constant_smooth)
        self.w = w

    def forward(self, logits, labels, **kwargs):
        bias_converted = self.bias_convert(**kwargs)
        loss = self.loss_compute(logits, labels, bias_converted, **kwargs)

        # Re-normalized version of the bias
        bias_norm = elementwise_logsumexp(bias_converted[:, :, 0], bias_converted[:, :, 1])
        bias_logprob = bias_converted - bias_norm.unsqueeze(2)

        # Compute and add the entropy penalty
        entropy = -(torch.exp(bias_logprob) * bias_logprob).sum(2).mean()
        return loss + self.w * entropy


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, epsilon=1e-9, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.reduction = reduction

    def forward(self, logits, labels, **kwargs):
        """
        logits: tensor of shape (N, num_answer)
        label: tensor of shape (N, num_answer)
        """
        logits = F.softmax(logits, dim=-1)
        ce_loss = - (labels * torch.log(logits)).sum(dim=-1)

        pt = torch.exp(-ce_loss)

        loss = self.alpha * (1 - pt)**self.gamma * ce_loss

        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        return loss
