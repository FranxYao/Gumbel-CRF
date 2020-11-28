"""A implementation of Linear-chain CRF inference algorithms, including:

* Viterbi, relaxed Viterbi
* Perturb and MAP sampling, and its relaxed version 
* Forward algorithm
* Entropy 
* Forward Filtering Backward Sampling, and it Gumbelized version

"""

import torch
import copy

import numpy as np 
import torch.nn as nn
import torch.nn.functional as F

from .. import torch_model_utils as tmu

class LinearChainCRF(nn.Module):
  """Implemention of the linear chain CRF, since we only need the forward, 
  relaxed sampling, and entropy here, we emit other inference algorithms like 
  forward backward, evaluation, and viterbi"""

  def __init__(self, config):
    super(LinearChainCRF, self).__init__()
    self.label_size = config.latent_vocab_size

    init_transition = torch.randn(
      self.label_size, self.label_size).to(config.device)

    # do not use any start or end index, assume that all links to start and end
    # has potential 1 
    self.transition = nn.Parameter(init_transition)
    return 

  def calculate_all_scores(self, emission_scores):
    """Mix the transition and emission scores

    Args:
      emission_scores: torch.Tensor(float), 
        size=[batch, max_len, num_class]

    Returns:
      scores: size=[batch, len, num_class, num_class]
        scores := log phi(batch, x_t, y_{t-1}, y_t)
    """
    label_size = self.label_size
    batch_size = emission_scores.size(0)
    seq_len = emission_scores.size(1)

    # scores[batch, t, C, C] = log_potential(t, from y_{t-1}, to y_t)
    scores = self.transition.view(1, 1, label_size, label_size)\
      .expand(batch_size, seq_len, label_size, label_size) + \
      emission_scores.view(batch_size, seq_len, 1, label_size)\
      .expand(batch_size, seq_len, label_size, label_size)

    return scores

  def forward_score(self, emission_scores, seq_lens):
    """The forward algorithm
    
    score = log(potential)

    Args:
      emission_scores: size=[batch, max_len, label_size]
      seq_lens: size=[batch]

    Returns:
      alpha: size=[batch, max_len, label_size]
      log_Z: size=[batch]
    """
    device = emission_scores.device
    all_scores = self.calculate_all_scores(emission_scores)

    batch_size = all_scores.size(0)
    seq_len = all_scores.size(1)
    alpha = torch.zeros(batch_size, seq_len, self.label_size).to(device)

    # the first position of all labels = 
    # (the transition from start - > all labels) + current emission.
    alpha[:, 0, :] = emission_scores[:, 0, :]

    for word_idx in range(1, seq_len):
      # batch_size, label_size, label_size
      before_log_sum_exp = alpha[:, word_idx - 1, :]\
        .view(batch_size, self.label_size, 1)\
        .expand(batch_size, self.label_size, self.label_size)\
        + all_scores[:, word_idx, :, :]
      alpha[:, word_idx, :] = torch.logsumexp(before_log_sum_exp, 1)

    # batch_size x label_size
    last_alpha = tmu.batch_gather_last(alpha, seq_lens)
    log_Z = torch.logsumexp(last_alpha, -1)
    return alpha, log_Z

  def backward_score(self, emission_scores, seq_lens):
    """backward algorithm
    
    Args:
      emission_scores: size=[batch, max_len, label_size]
      seq_lens: size=[batch]

    Returns:
      beta: size=[batch, max_len, num_class]
    """
    device = emission_scores.device
    all_scores = self.calculate_all_scores(emission_scores)

    batch_size = all_scores.size(0)
    seq_len = all_scores.size(1)

    # beta[T] initialized as 0
    beta = torch.zeros(batch_size, seq_len, self.label_size).to(device)

    # beta stored in reverse order
    # all score at i: phi(from class at L - i - 1, to class at L - i)
    all_scores = tmu.reverse_sequence(all_scores, seq_lens)
    for word_idx in range(1, seq_len):
      # beta[t + 1]: batch_size, t + 1, to label_size
      # indexing tricky here !! and different than the forward algo
      beta_t_ = beta[:, word_idx - 1, :]\
        .view(batch_size, 1, self.label_size)\
        .expand(batch_size, self.label_size, self.label_size)\

      # all_scores[t]: batch, from_state t-1, to state t
      before_log_sum_exp = beta_t_ + all_scores[:, word_idx - 1, :, :]
      beta[:, word_idx, :] = torch.logsumexp(before_log_sum_exp, 2)

    # reverse beta:
    beta = tmu.reverse_sequence(beta, seq_lens)
    # set the first beta to emission
    # beta[:, 0] = emission_scores[:, 0]
    return beta

  def log_prob(self, seq, emission_scores, seq_lens):
    """Evaluate the probability of a sequence
    
    Args:
      seq: size=[batch, max_len]
      emission_scores: size=[batch, max_len, num_class]
      seq_lens: size=[batch]

    Returns:
      log_prob: size=[batch]
    """
    device = emission_scores.device
    max_len = seq.size(1)
    batch_size = seq.size(0)
    alpha, log_Z = self.forward_score(emission_scores, seq_lens)
    score = torch.zeros(batch_size, max_len).to(device)
    
    for i in range(max_len):
      if(i == 0):
        score[:, i] += tmu.batch_index_select(
          emission_scores[:, 0], # [batch, num_class]
          seq[:, 0]) # [batch]
      else: 
        transition_ = self.transition.view(1, self.label_size, self.label_size)
        transition_ = transition_.repeat(batch_size, 1, 1)
        prev_ind = seq[:, i - 1] # [batch] 
        current_ind = seq[:, i] # [batch]
        # select prev index
        transition_ = tmu.batch_index_select(transition_, prev_ind)
        # select current index
        transition_ = tmu.batch_index_select(transition_, current_ind)
        score[:, i] += transition_
        score[:, i] += tmu.batch_index_select(emission_scores[:, i], current_ind)

    score = tmu.mask_by_length(score, seq_lens)
    log_prob = score.sum(-1) - log_Z
    return log_prob

  def marginal(self, seq, emission_scores, seq_lens):
    """Marginal distribution with conventional forward-backward

    TODO: an autograd based implementation 

    Args:
      seq: size=[batch, max_len]
      emission_scores: size=[batch, max_len, num_class]
      seq_lens: size=[batch]

    Returns:
      log_marginal: size=[batch, max_len]
    """
    alpha, log_Z = self.forward_score(emission_scores, seq_lens)
    beta = self.backward_score(emission_scores, seq_lens)

    # select weight according to the index to be evaluated
    batch_size = seq.size(0)
    max_len = seq.size(1)
    alpha_ = alpha.view(batch_size * max_len, -1)
    alpha_ = tmu.batch_index_select(alpha_, seq.view(-1))
    alpha_ = alpha_.view(batch_size, max_len)
    beta_ = beta.view(batch_size * max_len, -1)
    beta_ = tmu.batch_index_select(beta_, seq.view(-1))
    beta_ = beta_.view(batch_size, max_len)
    log_marginal = alpha_ + beta_ - log_Z.unsqueeze(1)
    return log_marginal

  def rargmax(self, emission_scores, seq_lens):
    """Relaxed Argmax from the CRF, Viterbi decoding.

    Everything is the same with pmsample except not using the gumbel noise

    Args:
      emission_scores: type=torch.tensor(float), 
        size=[batch, max_len, num_class]
      seq_lens: type=torch.tensor(int), size=[batch]

    Returns:
      y_hard: size=[batch, max_len]
      y: size=[batch, max_len, num_class]
    """
    device = emission_scores.device

    all_scores = self.calculate_all_scores(emission_scores)
    batch_size = all_scores.size(0)
    seq_len = all_scores.size(1)

    s = torch.zeros(batch_size, seq_len, self.label_size).to(device)
    s[:, 0] = emission_scores[:, 0] 

    # [B, T, from C, to C]
    bp = torch.zeros(batch_size, seq_len, self.label_size, self.label_size)
    bp = bp.to(device)
    
    # forward viterbi
    for t in range(1, seq_len):
      s_ = s[:, t - 1].unsqueeze(2) + all_scores[:, t] # [B, from C, to C]
      s[:, t] = s_.max(dim=1)[0] # [B, C]
      bp[:, t] = torch.softmax(s_ / tau, dim=1)

    # backtracking
    s = tmu.reverse_sequence(s, seq_lens)
    bp = tmu.reverse_sequence(bp, seq_lens)
    y = torch.zeros(batch_size, seq_len, self.label_size).to(device)
    y[:, 0] = torch.softmax(s[:, 0] / tau, dim=1)
    for t in range(1, seq_len):
      y_ = y[:, t-1].argmax(dim=1) # [B]
      y[:, t] = tmu.batch_index_select(
        bp[:, t-1].transpose(1, 2), # [B, to C, from C]
        y_ 
        )
    y = tmu.reverse_sequence(y, seq_lens)
    y_hard = y.argmax(dim=2)
    return y_hard, y

  def pmsample(self, emission_scores, seq_lens, tau):
    """Perturb-and-MAP sampling, a relaxed Viterbi with Gumbel-perturbation

    Args:
      emission_scores: type=torch.tensor(float), 
        size=[batch, max_len, num_class]
      seq_lens: type=torch.tensor(int), size=[batch]
      tau: type=float, anneal strength

    Returns
      sample: size=[batch, max_len]
      relaxed_sample: size=[batch, max_len, num_class]
    """

    device = emission_scores.device

    all_scores = self.calculate_all_scores(emission_scores)
    all_scores += tmu.sample_gumbel(all_scores.size()).to(device)
    batch_size = all_scores.size(0)
    seq_len = all_scores.size(1)

    s = torch.zeros(batch_size, seq_len, self.label_size).to(device)
    s[:, 0] = emission_scores[:, 0] 
    s[:, 0] += tmu.sample_gumbel(emission_scores[:, 0].size()).to(device)

    # [B, T, from C, to C]
    bp = torch.zeros(batch_size, seq_len, self.label_size, self.label_size)
    bp = bp.to(device)
    
    # forward viterbi
    for t in range(1, seq_len):
      s_ = s[:, t - 1].unsqueeze(2) + all_scores[:, t] # [B, from C, to C]
      s[:, t] = s_.max(dim=1)[0] # [B, C]
      bp[:, t] = torch.softmax(s_ / tau, dim=1)

    # backtracking
    s = tmu.reverse_sequence(s, seq_lens)
    bp = tmu.reverse_sequence(bp, seq_lens)
    y = torch.zeros(batch_size, seq_len, self.label_size).to(device)
    y[:, 0] = torch.softmax(s[:, 0] / tau, dim=1)
    for t in range(1, seq_len):
      y_ = y[:, t-1].argmax(dim=1) # [B]
      y[:, t] = tmu.batch_index_select(
        bp[:, t-1].transpose(1, 2), # [B, to C, from C]
        y_ )
    y = tmu.reverse_sequence(y, seq_lens)
    y_hard = y.argmax(dim=2)
    return y_hard, y

  def rsample(self, emission_scores, seq_lens, tau, 
    return_switching=False, return_prob=False):
    """Reparameterized CRF sampling, a Gumbelized version of the 
    Forward-Filtering Backward-Sampling algorithm

    TODO: an autograd based implementation 
    requires to redefine the backward function over a relaxed-sampling semiring
    
    Args:
      emission_scores: type=torch.tensor(float), 
        size=[batch, max_len, num_class]
      seq_lens: type=torch.tensor(int), size=[batch]
      tau: type=float, anneal strength

    Returns
      sample: size=[batch, max_len]
      relaxed_sample: size=[batch, max_len, num_class]
    """
    # Algo 2 line 1
    all_scores = self.calculate_all_scores(emission_scores)
    alpha, log_Z = self.forward_score(emission_scores, seq_lens) 

    batch_size = emission_scores.size(0)
    max_len = emission_scores.size(1)
    num_class = emission_scores.size(2)
    device = emission_scores.device

    # Backward sampling start
    # The sampling still goes backward, but for simple implementation we
    # reverse the sequence, so in the code it still goes from 1 to T 
    relaxed_sample_rev = torch.zeros(batch_size, max_len, num_class).to(device)
    sample_prob = torch.zeros(batch_size, max_len).to(device)
    sample_rev = torch.zeros(batch_size, max_len).type(torch.long).to(device)
    alpha_rev = tmu.reverse_sequence(alpha, seq_lens).to(device)
    all_scores_rev = tmu.reverse_sequence(all_scores, seq_lens).to(device)
    
    # Algo 2 line 3, log space
    # w.shape=[batch, num_class]
    w = alpha_rev[:, 0, :].clone()
    w -= log_Z.view(batch_size, -1)
    p = w.exp()
    # switching regularization for longer chunk, not mentioned in the paper
    # so do no need to care. In the future this will be updated with posterior
    # regularization
    if(return_switching): 
      switching = 0.
    
    # Algo 2 line 4
    relaxed_sample_rev[:, 0] = tmu.reparameterize_gumbel(w, tau)
    # Algo 2 line 5
    sample_rev[:, 0] = relaxed_sample_rev[:, 0].argmax(dim=-1)
    sample_prob[:, 0] = tmu.batch_index_select(p, sample_rev[:, 0]).flatten()
    mask = tmu.length_to_mask(seq_lens, max_len).type(torch.float)
    prev_p = p
    for i in range(1, max_len):
      # y_after_to_current[j, k] = log_potential(y_{t - 1} = k, y_t = j, x_t)
      # size=[batch, num_class, num_class]
      y_after_to_current = all_scores_rev[:, i-1].transpose(1, 2)
      # w.size=[batch, num_class]
      w = tmu.batch_index_select(y_after_to_current, sample_rev[:, i-1])
      w_base = tmu.batch_index_select(alpha_rev[:, i-1], sample_rev[:, i-1])
      # Algo 2 line 7, log space
      w = w + alpha_rev[:, i] - w_base.view(batch_size, 1)
      p = F.softmax(w, dim=-1) # p correspond to pi in the paper
      if(return_switching):
        switching += (tmu.js_divergence(p, prev_p) * mask[:, i]).sum()
      prev_p = p
      # Algo 2 line 8
      relaxed_sample_rev[:, i] = tmu.reparameterize_gumbel(w, tau)
      # Algo 2 line 9
      sample_rev[:, i] = relaxed_sample_rev[:, i].argmax(dim=-1)
      sample_prob[:, i] = tmu.batch_index_select(p, sample_rev[:, i]).flatten()

    # Reverse the sequence back
    sample = tmu.reverse_sequence(sample_rev, seq_lens)
    relaxed_sample = tmu.reverse_sequence(relaxed_sample_rev, seq_lens)
    sample_prob = tmu.reverse_sequence(sample_prob, seq_lens)
    sample_prob = sample_prob.masked_fill(mask == 0, 1.)
    sample_log_prob_stepwise = (sample_prob + 1e-10).log()
    sample_log_prob = sample_log_prob_stepwise.sum(dim=1)

    ret = [sample, relaxed_sample]
    if(return_switching): 
      switching /= (mask.sum(dim=-1) - 1).sum()
      ret.append(switching)
    if(return_prob):
      ret.extend([sample_log_prob, sample_log_prob_stepwise])
    return ret

  def entropy(self, emission_scores, seq_lens):
    """The entropy of the CRF, another DP algorithm. See the write up
    
    Args:
      emission_scores:
      seq_lens:

    Returns:
      H_total: the entropy, type=torch.Tensor(float), size=[batch]
    """

    all_scores = self.calculate_all_scores(emission_scores)
    alpha, log_Z = self.forward_score(emission_scores, seq_lens)

    batch_size = emission_scores.size(0)
    max_len = emission_scores.size(1)
    num_class = emission_scores.size(2)
    device = emission_scores.device

    H = torch.zeros(batch_size, max_len, num_class).to(device)
    for t in range(max_len - 1):
      # log_w.shape = [batch, from_class, to_class]
      log_w = all_scores[:, t+1, :, :] +\
        alpha[:, t, :].view(batch_size, num_class, 1) -\
        alpha[:, t+1, :].view(batch_size, 1, num_class)
      w = log_w.exp()
      H[:, t+1, :] = torch.sum(
        w * (H[:, t, :].view(batch_size, num_class, 1) - log_w), dim=1)
    
    last_alpha = tmu.gather_last(alpha, seq_lens)
    H_last = tmu.gather_last(H, seq_lens)
    log_p_T = last_alpha - log_Z.view(batch_size, 1)
    p_T = log_p_T.exp()

    H_total = p_T * (H_last - log_p_T)
    H_total = H_total.sum(dim = -1)
    return H_total