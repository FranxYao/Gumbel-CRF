

import torch
import copy

import numpy as np 
import torch.nn as nn
import torch.nn.functional as F

from .. import torch_model_utils as tmu

class LinearChainCRF(nn.Module):
  """"""
  def __init__(self, config):
    super(LinearChainCRF, self).__init__()
    self.label_size = config.latent_vocab_size
    self.device = config.device

    # TEST
    # init_transition = torch.Tensor([[1, 2, 3, 4], 
    #                                 [4, 3, 2, 1],
    #                                 [5, 6, 7, 8],
    #                                 [8, 7, 6, 5]])

    init_transition = torch.randn(
      self.label_size, self.label_size).to(config.device)
    self.start_id = 0
    self.end_id = self.label_size - 1
    # cannot transit to start index
    init_transition[:, self.start_id] = -1e10
    # end index do not transit to any index 
    init_transition[self.end_id, :] = -1e10
    self.transition = nn.Parameter(init_transition)
    return 

  def calculate_all_scores(self, emission_scores):
    """Mix the transition and emission scores

    Args:
      emission_scores: type=torch.Tensor(float), 
        size=[batch, max_len, num_class]

    Returns:
      scores: size=[batch, len, num_class, num_class]
      scores = log phi(batch, x_t, y_{t-1}, y_t)
    """
    # mask out the emission: the current emission cannot from start or end
    emission_scores[:, :, self.start_id] = -1e10
    emission_scores[:, :, self.end_id] = -1e10

    label_size = self.label_size
    batch_size = emission_scores.size(0)
    seq_len = emission_scores.size(1)
    scores = self.transition.view(1, 1, label_size, label_size)\
      .expand(batch_size, seq_len, label_size, label_size) + \
      emission_scores.view(batch_size, seq_len, 1, label_size)\
      .expand(batch_size, seq_len, label_size, label_size)
    # from the second step, the current state cannot be from the start state
    scores[:, 1:, self.start_id, :] = -1e10
    return scores

  def forward_score(self, all_scores, sent_lens):
    """The forward algorithm
    
    score = log(potential)

    Returns:
      alpha: size=[batch, max_len, label_size]
      Z: size=[batch]
    """
    batch_size = all_scores.size(0)
    seq_len = all_scores.size(1)
    alpha = torch.zeros(batch_size, seq_len, self.label_size).to(self.device)

    # the first position of all labels = 
    # (the transition from start - > all labels) + current emission.
    alpha[:, 0, :] = all_scores[:, 0,  self.start_id, :] 

    for word_idx in range(1, seq_len):
      # batch_size, label_size, label_size
      before_log_sum_exp = alpha[:, word_idx - 1, :]\
        .view(batch_size, self.label_size, 1)\
        .expand(batch_size, self.label_size, self.label_size)\
        + all_scores[:, word_idx, :, :]
      alpha[:, word_idx, :] = tmu.logsumexp(before_log_sum_exp)

    # batch_size x label_size
    last_alpha = torch.gather(alpha, 1, sent_lens.view(batch_size, 1, 1)\
      .expand(batch_size, 1, self.label_size) - 1)\
        .view(batch_size, self.label_size)
    last_alpha += self.transition[:, self.end_id]\
      .view(1, self.label_size).expand(batch_size, self.label_size)
    # batch_size
    last_alpha = tmu.logsumexp(
      last_alpha.view(batch_size, self.label_size, 1)).view(batch_size)
    log_Z = last_alpha
    return alpha, log_Z

  def argmax(self, emission_scores, sent_lens):
    """Argmax from the CRF, Viterbi decoding"""
    return 

  def rsample(self, emission_scores, sent_lens, tau):
    """reparameterized sampling, a gumbelized version of the forward-filtering
    backward-sampling algorithm

    TODO: an autograd based implementation
    
    Args:
      emission_scores: type=torch.tensor(float), 
        size=[batch, max_len, num_class]
      sent_lens: type=torch.tensor(int), size=[batch]

    Returns
      sample: size=[batch, max_len]
      relaxed_sample: size=[batch, max_len, num_class]
    """
    all_scores = self.calculate_all_scores(emission_scores)
    alpha, log_Z = self.forward_score(all_scores, sent_lens)

    batch_size = emission_scores.size(0)
    max_len = emission_scores.size(1)
    num_class = emission_scores.size(2)
    device = self.device

    # backward filtering, reverse the sequence
    relaxed_sample_rev = torch.zeros(batch_size, max_len, num_class)
    sample_rev = torch.zeros(batch_size, max_len).type(torch.long).to(device)
    alpha_rev = tmu.reverse_sequence(alpha, sent_lens)
    all_scores_rev = tmu.reverse_sequence(all_scores, sent_lens)
    
    # w.shape=[batch, num_class]
    w = self.transition[:, self.end_id].view(1, num_class) + alpha_rev[:, 0, :]
    w -= log_Z.view(batch_size, -1)

    # DEBUG, to show exp(w) gives a valid distribution
    # p(y_T = k | x) = exp(w)
    # print(torch.exp(w)[0])
    # print(torch.exp(w)[0].sum())
    
    relaxed_sample_rev[:, 0] = tmu.reparameterize_gumbel(w, tau)
    sample_rev[:, 0] = relaxed_sample_rev[:, 0].argmax(dim=-1)
    for i in range(1, max_len):
      # y_after_to_current[j, k] = log_potential(y_{t - 1} = k, y_t = j, x_t)
      # size=[batch, num_class, num_class]
      y_after_to_current = all_scores_rev[:, i-1].transpose(1, 2)
      # w.size=[batch, num_class]
      w = tmu.batch_index_select(y_after_to_current, sample_rev[:, i-1])
      w_base = tmu.batch_index_select(alpha_rev[:, i-1], sample_rev[:, i-1])
      w = w + alpha_rev[:, i] - w_base.view(batch_size, 1)

      # DEBUG: to show exp(w) gives a valid distribution
      # p(y_{t - 1} = j | y_t = k, x) = exp(w)
      # print(torch.exp(w)[0])
      # print(torch.exp(w)[0].sum())
      
      relaxed_sample_rev[:, i] = tmu.reparameterize_gumbel(w, tau)
      sample_rev[:, i] = relaxed_sample_rev[:, i].argmax(dim=-1)

    sample = tmu.reverse_sequence(sample_rev, sent_lens)
    relaxed_sample = tmu.reverse_sequence(relaxed_sample_rev, sent_lens)
    return sample, relaxed_sample

  def entropy(self, emission_scores):
    """The entropy of the CRF"""
    return 