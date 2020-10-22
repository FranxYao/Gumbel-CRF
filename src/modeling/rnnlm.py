
import copy

import numpy as np 

import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.optim import Adam, SGD, RMSprop

from .lstm_seq2seq.encoder import LSTMEncoder
from .ftmodel import FTModel

class RNNLM(nn.Module):

  def __init__(self, config):
    super().__init__()

    self.pad_id = config.pad_id
    self.start_id = config.start_id
    self.end_id = config.end_id
    self.vocab_size = config.vocab_size
    self.lstm_layers = config.lstm_layers

    self.embeddings = nn.Embedding(config.vocab_size, config.embedding_size)
    config.lstm_bidirectional = False
    self.encoder = LSTMEncoder(config)
    self.proj = nn.Linear(config.state_size, config.vocab_size)
    self.proj.weight = self.embeddings.weight
    return 

  def forward(self, sentences, sent_lens):
    """
    Args:
    
    Returns:
    """
    sent_inputs = sentences[:, :-1]
    sent_targets = sentences[:, 1:]
    out_dict = {}
    batch_size = sent_inputs.size(0)
    mask = sent_inputs != self.pad_id

    sent_inputs_emb = self.embeddings(sent_inputs)
    enc_states, (_, _) = self.encoder(sent_inputs_emb, sent_lens)
    logits = self.proj(enc_states)

    max_len = logits.size(1)
    mask = mask[:, :max_len]

    log_prob = -F.cross_entropy(logits.view(-1, self.vocab_size), 
      sent_targets[:, :max_len].flatten(), reduction='none')
    log_prob = log_prob.view(batch_size, max_len)
    log_prob = log_prob.masked_fill(mask == 0, 0.)
    nll = log_prob.sum(1).mean(0)
    out_dict['nll'] = nll.item()
    log_prob = log_prob.sum() / mask.sum().type(torch.float)

    loss = -log_prob
    out_dict['loss'] = loss.detach().cpu().numpy()
    out_dict['ppl'] = (-log_prob).exp().detach().cpu().numpy()
    out_dict['neg_ppl'] = -out_dict['ppl']
    return loss, out_dict

  def infer(self, sentences):
    return 


class RNNLMModel(FTModel):

  def __init__(self, config):
    super().__init__()

    self.model = RNNLM(config)
    self.optimizer = Adam(self.model.parameters(), lr=config.learning_rate)
    self.task = config.task 
    self.dataset = config.dataset
    self.device = config.device
    self.max_grad_norm = config.max_grad_norm
    return 

  def train_step(self, batch, n_iter, ei, bi, schedule_params):
    model = self.model
    model.zero_grad()
    loss, out_dict = model(
      sentences=torch.from_numpy(batch['sent_dlex']).to(self.device),
      sent_lens=torch.from_numpy(batch['sent_lens']).to(self.device)
      )
    loss.backward()
    clip_grad_norm_(model.parameters(), self.max_grad_norm)
    self.optimizer.step()
    return out_dict

  def valid_step(self, template_manager, batch, n_iter, ei, bi, 
    mode='dev', dataset=None, schedule_params=None):
    model = self.model
    _, out_dict = model(
      sentences=torch.from_numpy(batch['sent_dlex']).to(self.device),
      sent_lens=torch.from_numpy(batch['sent_lens']).to(self.device)
      )
    return out_dict