
import copy

import numpy as np 

import torch 
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

from .lstm_seq2seq.encoder import LSTMEncoder
from .lstm_seq2seq.decoder import LSTMDecoder
from . import torch_model_utils as tmu

class GaussianVAE(nn.Module):

  def __init__(self, config):
    super(GaussianVAE, self).__init__()

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
    self.z_proj = nn.Linear(config.state_size, 2 * config.state_size)
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

  def forward(self, sentences, sent_lens):
    """"""
    out_dict = {}
    inspect = {}
    loss = 0.0

    # encoder
    batch_size = sentences.size(0)
    sent_emb = self.embeddings(sentences)
    _, (sent_enc, _) = self.encoder(sent_emb, sent_lens)

    # latent 
    z_enc = self.z_proj(sent_enc[-1])
    z_mu = z_enc[:, :self.state_size].clone()
    z_logvar = z_enc[:, self.state_size :].clone()
    z = tmu.reparameterize_gaussian(z_mu, z_logvar)

    # KLD loss 
    kld =  -0.5 * torch.sum(1 + z_logvar - z_mu.pow(2) - z_logvar.exp())
    loss += self.z_beta * kld
    out_dict['kld'] = tmu.to_np(kld)

    # decoder init
    init_state = self.init_state(z)
    mem, mem_emb, mem_mask = None, None, None
    dec_inputs = self.embeddings(sentences[:, :-1])

    # # residual connections
    # max_len = dec_inputs.size(1)
    # dec_inputs += z.view(batch_size, 1, state_size).repeat(1, max_len, 1)

    # decoding
    dec_targets = sentences[:, 1:]
    log_prob, predictions = self.decoder.decode_train(init_state, 
      mem, mem_emb, mem_mask, dec_inputs, dec_targets)
    loss -= log_prob
    out_dict['log_prob'] = tmu.to_np(log_prob)
    out_dict['loss'] = tmu.to_np(loss)

    return loss, out_dict

  def infer(self, sentences, sent_lens, num_sample):
    """"""
    out_dict = {}

    # encoder
    batch_size = sentences.size(0)
    sent_emb = self.embeddings(sentences)
    # print('debug')
    # print(sent_emb.shape)
    # print(sent_lens.shape)
    # print(sent_lens)
    _, (sent_enc, _) = self.encoder(sent_emb, sent_lens)

    # latent 
    z_enc = self.z_proj(sent_enc[-1])
    z_mu = z_enc[:, :self.state_size].clone()
    z_logvar = z_enc[:, self.state_size :].clone()
    z_mu = z_mu.view(batch_size, 1, self.state_size).repeat(1, num_sample, 1)
    z_mu = z_mu.view(batch_size * num_sample, self.state_size)
    z_logvar = z_logvar.view(
      batch_size, 1, self.state_size).repeat(1, num_sample, 1)
    z_logvar = z_logvar.view(batch_size * num_sample, self.state_size)
    z = tmu.reparameterize_gaussian(z_mu, z_logvar)

    # decoding
    init_state = self.init_state(z)
    mem, mem_emb, mem_mask = None, None, None
    predictions_all = self.decoder.decode_infer(init_state, 
      self.embeddings, mem, mem_emb, mem_mask)
    max_dec_len = predictions_all.size(1)
    predictions_all = predictions_all.view(batch_size, num_sample, max_dec_len)
    predictions = predictions_all[:, 0]
    
    out_dict['predictions_all'] = tmu.to_np(predictions_all)
    out_dict['predictions'] = tmu.to_np(predictions)
    return out_dict
