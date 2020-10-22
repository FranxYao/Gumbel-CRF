"""Sequence to sequence baseline model"""

import copy

import numpy as np 

import torch 
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

from .lstm_seq2seq.encoder import LSTMEncoder
from .lstm_seq2seq.decoder import LSTMDecoder
from . import torch_model_utils as tmu

class Seq2seq(nn.Module):

  def __init__(self, config):
    super(Seq2seq, self).__init__()

    self.config = config
    self.pad_id = config.pad_id
    self.start_id = config.start_id
    self.end_id = config.end_id

    self.z_beta = config.z_beta

    self.lstm_layers = config.lstm_layers
    self.embedding_size = config.embedding_size
    self.state_size = config.state_size
    self.max_dec_len = config.max_dec_len

    self.embeddings = nn.Embedding(config.vocab_size, config.embedding_size)
    self.encoder = LSTMEncoder(config)
    config.copy_decoder = True # always use copy decoder 
    self.decoder = LSTMDecoder(config)
    self.dec_init_state_proj_h = nn.Linear(
      config.embedding_size, config.lstm_layers * config.state_size)
    self.dec_init_state_proj_c = nn.Linear(
      config.embedding_size, config.lstm_layers * config.state_size)
    return 

  def init_state(self, s):
    batch_size = s.shape[0]
    init_state_h = self.dec_init_state_proj_h(s)
    init_state_h = init_state_h.view(
      batch_size, self.lstm_layers, self.state_size)
    init_state_h = init_state_h.transpose(0, 1).contiguous()
    init_state_c = self.dec_init_state_proj_c(s)
    init_state_c = init_state_c.view(
      batch_size, self.lstm_layers, self.state_size)
    init_state_c = init_state_c.transpose(0, 1).contiguous()
    return (init_state_h, init_state_c)

  def forward(self, enc_inputs, enc_lens, dec_inputs, dec_targets):
    """"""
    out_dict = {}
    inspect = {}
    loss = 0.0

    # encoder
    batch_size = enc_inputs.size(0)
    sent_emb = self.embeddings(enc_inputs)
    sent_enc, enc_state = self.encoder(sent_emb, enc_lens)

    # decoder init
    max_len = sent_enc.size(1)
    mem = enc_inputs[:, :max_len]
    mem_emb = sent_enc[:, :max_len]
    mem_mask = tmu.length_to_mask(enc_lens, max_len)
    dec_inputs = self.embeddings(dec_inputs)

    # decoding
    log_prob, predictions = self.decoder.decode_train(enc_state, 
      mem, mem_emb, mem_mask, dec_inputs, dec_targets)
    loss += log_prob
    out_dict['log_prob'] = tmu.to_np(log_prob)

    loss = -loss
    out_dict['loss'] = tmu.to_np(loss)
    return loss, out_dict

  def infer_prev(self, enc_inputs, enc_lens, num_sample=2):
    """"""
    out_dict = {}

    # encoder
    batch_size = enc_inputs.size(0)
    sent_emb = self.embeddings(enc_inputs)
    sent_enc, enc_state = self.encoder(sent_emb, enc_lens)

    # decoder init
    max_len = sent_enc.size(1)
    mem = enc_inputs[:, :max_len]
    mem_emb = sent_enc[:, :max_len]
    mem_mask = tmu.length_to_mask(enc_lens, max_len)

    # decoding
    predictions_all = self.decoder.decode_infer(enc_state, 
      self.embeddings, mem, mem_emb, mem_mask)
    max_dec_len = predictions_all.size(1)
    predictions_all = predictions_all.view(-1, num_sample, max_dec_len)
    predictions = predictions_all[:, 0]
    
    out_dict['predictions_all'] = tmu.to_np(predictions_all)
    out_dict['predictions'] = tmu.to_np(predictions)
    return out_dict

  def infer_sample(self, enc_inputs, enc_lens, num_sample=3):
    out_dict = {}

    # encoder
    batch_size = enc_inputs.size(0)
    enc_inputs = enc_inputs.view(batch_size, 1, -1).repeat(1, num_sample, 1)
    enc_inputs = enc_inputs.view(batch_size * num_sample, -1)
    enc_lens = enc_lens.view(batch_size, 1).repeat(1, num_sample)
    enc_lens = enc_lens.view(batch_size * num_sample)

    sent_emb = self.embeddings(enc_inputs)
    sent_enc, enc_state = self.encoder(sent_emb, enc_lens)

    # decoder init
    max_len = sent_enc.size(1)
    mem = enc_inputs[:, :max_len]
    mem_emb = sent_enc[:, :max_len]
    mem_mask = tmu.length_to_mask(enc_lens, max_len)

    # decoding 
    predictions_all = self.decoder.decode_infer(enc_state, 
      self.embeddings, mem, mem_emb, mem_mask)
    max_dec_len = predictions_all.size(1)
    predictions_all = predictions_all.view(-1, num_sample, max_dec_len)
    predictions = predictions_all[:, 0]

    out_dict['predictions_all'] = tmu.to_np(predictions_all)
    out_dict['predictions'] = tmu.to_np(predictions)
    return out_dict
