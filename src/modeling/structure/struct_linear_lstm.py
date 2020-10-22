"""Latent variable structured prediction """

import torch
import copy

import numpy as np 
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions import Categorical
from ..torch_model_utils import ind_to_one_hot
from ..lstm_seq2seq.decoder import LSTMDecoder

class StructLinearLSTM(nn.Module):
  """Linear chain structure parameterized by LSTM.
  
  View LSTM as a probabilistic model that supports evaluation, sampling, greedy, 
  and beam search decoding. 
  """

  def __init__(self, config, embeddings=None):
    """Initialization"""
    super(StructLinearLSTM, self).__init__()

    self.lstm_layers = config.lstm_layers
    self.state_size = config.state_size
    self.max_dec_len = config.max_dec_len
    self.start_id = config.latent_vocab_size - 1

    self.latent_embeddings = nn.Embedding(
      config.latent_vocab_size, config.embedding_size)
    self.dec_init_state_proj_h = nn.Linear(
      config.embedding_size, config.lstm_layers * config.state_size)
    self.dec_init_state_proj_c = nn.Linear(
      config.embedding_size, config.lstm_layers * config.state_size)
    self.decoder = LSTMDecoder(config)
    return 

  def init_state(self, y):
    """Init decoding state

    Args:
      y: torch.tensor(torch.Float), size=[batch, state_size]
    """
    batch_size = y.shape[0]
    init_state_h = self.dec_init_state_proj_h(y)
    init_state_h = init_state_h.view(
      batch_size, self.lstm_layers, self.state_size)
    init_state_h = init_state_h.transpose(0, 1)
    init_state_c = self.dec_init_state_proj_c(y)
    init_state_c = init_state_c.view(
      batch_size, self.lstm_layers, self.state_size)
    init_state_c = init_state_c.transpose(0, 1)
    return (init_state_h, init_state_c)

  def sample_step(self, probs, strategy='unconstrained', k=3):
    """Sampling in single decoding step

    Args:
      probs: type=torch.tensor(torch.Float), size=[batch, vocab] or 
        [batch * num_sample, vocab] 

    Returns:
      sample_index: type=torch.tensor(torch.int), size=[batch] or 
        [batch * num_sample], not differentiable
      sample_log_prob: type=torch.tensor(torch.float), size=[batch] or 
        [batch * num_sample], differentiable
    """
    if(strategy == 'unconstrained'):
      m = Categorical(probs=probs)
      sample_index = m.sample()
      sample_log_prob = m.log_prob(sample_index)
    elif(strategy == 'greedy'):
      sample_prob, sample_index = torch.max(probs, 1)
      sample_log_prob = torch.log(sample_prob)
    elif(strategy == 'topk'):
      topk_prob, topk_index = torch.topk(probs, k=k, dim=-1)
      # topk_prob.size = [batch, k]
      topk_prob /= topk_prob.sum(dim=-1, keepdim=True)
      m = Categorical(probs=topk_prob)
      sample_index = m.sample()
      sample_log_prob = m.log_prob(sample_index)
      # topk_index.size = [batch, k], sample_one_hot.size = [batch, k]
      sample_one_hot = ind_to_one_hot(sample_index, k, probs.device)
      sample_index = torch.masked_select(topk_index, sample_one_hot)
    else: 
      raise NotImplementedError('sample strategy %s not implemented' % strategy)
    return sample_index, sample_log_prob

  def sample_conditional(self, y, kv_emb, kv_mask, z_mask, 
    num_sample=1, strategy='unconstrained', gumbel=False, tau=1.):
    """Hard sample
    Args:
      see sample_conditional_hard and sample_conditional_gumbel

    Returns: 
      z_sample_ids: type=torch.tensor(torch.int), 
        shape=[batch, num_sample, max_len] 
      z_sample_states: type=torch.tensor(torch.float), 
        shape=[batch, num_sample, max_len, state]
      z_sample_log_prob: type=torch.tensor(torch.float), 
        shape=[batch, num_sample]
      z_sample_log_prob_stepwise: type=torch.tensor(torch.float)
        shape=[batch, num_sample, max_len]
      inspect: ispection, for monitoring training and debugging
    """
    if(gumbel):
      return self.sample_conditional_gumbel(
        y, kv_emb, kv_mask, z_mask, num_sample=num_sample, tau=tau)
    else: 
      return self.sample_conditional_hard(
        y, kv_emb, kv_mask, z_mask, num_sample=num_sample, strategy=strategy)

  def sample_conditional_hard(self, y, kv_emb, kv_mask, z_mask, 
    num_sample=1, strategy='unconstrained', k=3):
    """hard sampling, paralleled implementation

    Args:
      y: torch.tensor(torch.Float), size=[batch, state]
      kv_emb: torch.tensor(torch.float), size=[batch, mem_len, state]
      kv_mask: torch.tensor(torch.bool), size=[batch, mem_len]
      z_mask: torch.tensor(torch.float), size=[batch, max_len], 1 = not masked, 
        0 = masked
      num_sample: number of sample
      strategy: 'unconstrained', 'greedy', 'top-k'

    Returns:
      z_sample_ids: type=torch.tensor(torch.int), 
        shape=[batch, num_sample, max_len] 
      z_sample_states: type=torch.tensor(torch.float), 
        shape=[batch, num_sample, max_len, state]
      z_sample_log_prob: type=torch.tensor(torch.float), 
        size=[batch, num_sample]
      z_sample_log_prob_stepwise: type=torch.tensor(torch.float), 
        size=[batch, num_sample, max_len]
    """
    inspect = {}

    device = y.device
    batch_size = y.shape[0]
    state_size = self.state_size
    max_dec_len = self.max_dec_len
    mem_len = kv_emb.shape[1]

    embeddings = self.latent_embeddings
    dec_cell = self.decoder
    dec_proj = self.decoder.output_proj

    y = y.view(batch_size, 1, state_size).repeat(1, num_sample, 1)
    y = y.view(batch_size * num_sample, state_size)
    kv_emb = kv_emb.view(batch_size, 1, mem_len, state_size)
    kv_emb = kv_emb.repeat(1, num_sample, 1, 1)
    kv_emb = kv_emb.view(batch_size * num_sample, mem_len, state_size)
    kv_mask = kv_mask.view(batch_size, 1, mem_len)
    kv_mask = kv_mask.repeat(1, num_sample, 1)
    kv_mask = kv_mask.view(batch_size * num_sample, mem_len)
    z_mask = z_mask.view(batch_size, 1, max_dec_len)
    z_mask = z_mask.repeat(1, num_sample, 1)

    state = self.init_state(y)
    inp = embeddings(torch.zeros(batch_size * num_sample, 
      dtype=torch.long).to(device) + self.start_id)

    # output
    z_sample = []
    z_sample_log_prob = []
    # inspection
    latent_state_ent = []
    latent_state_vocab = []
    latent_state_vocab_ent = []
    for i in range(max_dec_len):
      # out.shape = [time_step=1, batch * num_sample, state_size], 
      out, state = dec_cell(inp, state, kv_emb, kv_mask)
      prob = F.softmax(dec_proj(out)[0], dim=-1) # [batch * sample, vocab]
      out_index, sample_log_prob = self.sample_step(prob, strategy, k)
      z_sample.append(out_index)
      z_sample_log_prob.append(sample_log_prob)
      inp = embeddings(out_index)

      # inspection
      ent_ = (-prob * torch.log(prob)).sum(dim=1).mean()
      latent_state_ent.append(ent_)

    # output
    z_sample = torch.stack(z_sample).transpose(1, 0)
    z_sample_ids = z_sample.view(batch_size, num_sample, max_dec_len)
    inspect['z_sample_ids'] = z_sample_ids.detach().cpu().numpy()
    z_sample_states = embeddings(z_sample_ids)
    z_sample_log_prob = torch.stack(z_sample_log_prob).transpose(1, 0)
    z_sample_log_prob_stepwise = z_sample_log_prob.view(
      batch_size, num_sample, max_dec_len)
    z_sample_log_prob = (z_sample_log_prob_stepwise * z_mask).sum(dim=2)

    # inspection
    latent_state_ent = torch.stack(latent_state_ent).mean()
    ent_z = latent_state_ent
    inspect['latent_state_ent'] = latent_state_ent.detach().cpu().numpy()
    return (z_sample_ids, z_sample_states, z_sample_log_prob, 
      z_sample_log_prob_stepwise, ent_z,inspect)

  def sample_conditional_gumbel(self, y, kv_emb, kv_mask, z_mask, 
    num_sample=1, tau=1.):
    """Reparameterized sampling"""
    return 
  
  def sample_prior(self):
    return

  def eval_conditional(self, y):
    return 

  def greedy_conditional(self, y):

    return 

  def sample_prior(self):
    return 
