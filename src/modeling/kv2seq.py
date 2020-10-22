
import copy

import numpy as np 

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

from torch.distributions import Categorical

from .lstm_seq2seq.encoder import LSTMEncoder
from .lstm_seq2seq.decoder import LSTMDecoder
from . import torch_model_utils as tmu

class KV2Seq(nn.Module):
  """The key-value to sequence model"""

  def __init__(self, config, embeddings=None):
    super(KV2Seq, self).__init__()

    self.config = config

    self.pad_id = config.pad_id
    self.start_id = config.start_id
    self.end_id = config.end_id

    self.lstm_layers = config.lstm_layers
    self.embedding_size = config.embedding_size
    self.state_size = config.state_size
    self.max_dec_len = config.max_dec_len

    self.embeddings = nn.Embedding(config.vocab_size, config.embedding_size)
    self.decoder = LSTMDecoder(config)
    self.p_dec_init_state_proj_h = nn.Linear(
      config.embedding_size, config.lstm_layers * config.state_size)
    self.p_dec_init_state_proj_c = nn.Linear(
      config.embedding_size, config.lstm_layers * config.state_size)

    return 

  def init_state(self, s):
    batch_size = s.shape[0]
    init_state_h = self.p_dec_init_state_proj_h(s)
    init_state_h = init_state_h.view(
      batch_size, self.lstm_layers, self.state_size)
    init_state_h = init_state_h.transpose(0, 1).contiguous()
    init_state_c = self.p_dec_init_state_proj_c(s)
    init_state_c = init_state_c.view(
      batch_size, self.lstm_layers, self.state_size)
    init_state_c = init_state_c.transpose(0, 1).contiguous()
    return (init_state_h, init_state_c)

  def encode_kv(self, keys, vals):
    """Encode the key-valud table"""
    kv_mask = keys != self.pad_id 
    keys_emb = self.embeddings(keys)
    vals_emb = self.embeddings(vals)
    kv_emb = keys_emb + vals_emb # [batch, mem_len, state_size]

    kv_mask_ = kv_mask.type(torch.float)
    kv_enc = kv_emb * kv_mask_.unsqueeze(-1)
    # kv_enc.shape = [batch, embedding_size]
    kv_enc = kv_enc.sum(dim=1) / kv_mask_.sum(dim=1, keepdim=True)
    return kv_emb, kv_enc, kv_mask

  def forward(self, keys, vals, sentences):
    """Forward pass training"""
    out_dict = {}
    inspect = {}

    # encoder
    kv_emb, kv_enc, kv_mask = self.encode_kv(keys, vals)

    # decoder
    # log_prob = self.decode_train(kv_emb, kv_enc, kv_mask, sentences)
    init_state = self.init_state(kv_enc)
    dec_inputs = self.embeddings(sentences[:, :-1])
    log_prob, _ = self.decoder.decode_train(init_state, 
      keys, kv_emb, kv_mask, dec_inputs, sentences[:, 1:])
    # loss += log_prob
    # out_dict['log_prob'] = tmu.to_np(log_prob)

    loss = -log_prob
    out_dict['p_log_prob'] = tmu.to_np(log_prob)
    out_dict['loss'] = tmu.to_np(loss)
    return loss, out_dict

  def decode_train(self, mem_emb, mem_enc, mem_mask, sentences):
    """Decoder training loop"""
    batch_size = mem_emb.size(0)

    state = self.init_state(mem_enc)
    dec_inputs = self.embeddings(sentences[:, :-1])
    dec_targets = sentences[:, 1:]
    max_dec_len = dec_targets.size(1)

    dec_inputs = dec_inputs.transpose(0, 1)
    dec_targets = dec_targets.transpose(0, 1)
    dec_cell = self.decoder
    log_prob = []
    for i in range(max_dec_len):
      dec_out, state = dec_cell(
        dec_inputs[i], state, mem_emb, mem_mask)
      dec_out = dec_out[0]
      logits = dec_cell.output_proj(dec_out)

      log_prob_i = -F.cross_entropy(logits, dec_targets[i], reduction='none')
      log_prob.append(log_prob_i)

    log_prob = torch.stack(log_prob) # [T, B]
    mask = dec_targets != self.pad_id
    log_prob.masked_fill_(mask == 0, 0.) 
    log_prob = log_prob.sum() / mask.sum()
    return log_prob

  def infer(self, keys, vals, strategy='greedy'):
    """Inference"""
    out_dict = {}

    # encoder
    kv_emb, kv_enc, kv_mask = self.encode_kv(keys, vals)

    # decoder
    predictions, inspect = self.decode_infer(kv_emb, kv_enc, kv_mask, strategy)
    pred_lens = tmu.seq_ends(predictions[:, :-1], self.end_id) + 1

    out_dict['predictions'] = tmu.to_np(predictions)
    out_dict['pred_lens'] = tmu.to_np(pred_lens)
    return out_dict

  def infer_sample(self, keys, vals, num_sample=3):
    """Inference, Sampling based decoding"""
    out_dict = {}
    batch_size = keys.size(0)
    max_mem_len = keys.size(1)

    # encoder
    kv_emb, kv_enc, kv_mask = self.encode_kv(keys, vals)
    
    # decoder
    kv_enc = kv_enc.view(batch_size, 1, -1).repeat(1, num_sample, 1)
    kv_enc = kv_enc.view(batch_size * num_sample, -1)
    init_state = self.init_state(kv_enc)
    keys = keys.view(batch_size, 1, -1).repeat(1, num_sample, 1)
    keys = keys.view(batch_size * num_sample, -1)
    kv_emb = kv_emb.view(batch_size, 1, max_mem_len, -1)
    kv_emb = kv_emb.repeat(1, num_sample, 1, 1)
    kv_emb = kv_emb.view(batch_size * num_sample, max_mem_len, -1)
    kv_mask = kv_mask.view(batch_size, 1, -1).repeat(1, num_sample, 1)
    kv_mask = kv_mask.view(batch_size * num_sample, -1)

    predictions_all = self.decoder.decode_infer(init_state, self.embeddings,
      keys, kv_emb, kv_mask)
    max_dec_len = predictions_all.size(1)
    predictions_all = predictions_all.view(-1, num_sample, max_dec_len)
    predictions = predictions_all[:, 0]

    out_dict['predictions_all'] = tmu.to_np(predictions_all)
    out_dict['predictions'] = tmu.to_np(predictions)
    return out_dict

  def decode_infer(self, mem_emb, mem_enc, mem_mask, strategy='greedy'):
    """Inference decoding"""
    inspect = {}

    batch_size = mem_emb.size(0)
    device = mem_emb.device

    state = self.init_state(mem_enc)
    inp = torch.zeros(batch_size).to(device) + self.start_id
    inp = self.embeddings(inp.type(torch.long))

    dec_outputs = []
    dec_cell = self.decoder
    for i in range(self.max_dec_len):
      dec_out, state = dec_cell(inp, state, mem_emb, mem_mask)
      dec_out = dec_out[0]
      logits = dec_cell.output_proj(dec_out)
      if(strategy == 'greedy'):
        log_prob_i, out_index = torch.max(logits, 1)
      elif(strategy == 'sampling_unconstrained'):
        dist = Categorical(logits=logits)
        out_index = dist.sample()
      elif(strategy == 'sampling_top_k'):
        raise NotImplementedError(
          'decoding strategy %s not implemented' % strategy)
      elif(strategy == 'sampling_top_p'):
        raise NotImplementedError(
          'decoding strategy %s not implemented' % strategy)
      else: 
        raise NotImplementedError(
          'decoding strategy %s not implemented' % strategy)
      inp = self.embeddings(out_index)

      dec_outputs.append(out_index)

    dec_outputs = torch.stack(dec_outputs).transpose(1, 0) # [B, T]
    return dec_outputs, inspect