
import copy

import numpy as np 

import torch 
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

from .lstm_seq2seq.encoder import LSTMEncoder
from .lstm_seq2seq.decoder import LSTMDecoder, Attention
from . import torch_model_utils as tmu
from . import torch_struct

class LatentTemplateCRFTS(nn.Module):
  """The latent template model, CRF version, table to text setting"""

  def __init__(self, config, embeddings=None):
    super(LatentTemplateCRFTS, self).__init__()
    self.config = config

    self.z_beta = config.z_beta
    self.z_gamma = config.z_gamma
    self.gumbel_st = config.gumbel_st
    self.dec_adaptive = config.dec_adaptive
    self.use_copy = config.use_copy

    self.pad_id = config.pad_id
    self.start_id = config.start_id
    self.end_id = config.end_id
    self.seg_id = config.seg_id

    self.vocab_size = config.vocab_size
    self.latent_vocab_size = config.latent_vocab_size 

    self.lstm_layers = config.lstm_layers
    self.embedding_size = config.embedding_size
    self.state_size = config.state_size
    self.max_dec_len = config.max_dec_len
    self.max_bow_len = config.max_bow_len

    self.embeddings = nn.Embedding(config.vocab_size, config.embedding_size)
    if(embeddings is not None): 
      self.embeddings.weight.data.copy_(torch.from_numpy(embeddings))
    self.z_embeddings = nn.Embedding(
      config.latent_vocab_size, config.embedding_size)

    self.q_encoder = LSTMEncoder(config)

    self.z_crf_proj = nn.Linear(config.state_size, config.latent_vocab_size)
    init_transition = torch.randn(
      config.latent_vocab_size, config.latent_vocab_size).to(config.device)
    self.z_crf_transition = nn.Parameter(init_transition)

    self.p_dec_init_state_proj_h = nn.Linear(
      config.embedding_size, config.lstm_layers * config.state_size)
    self.p_dec_init_state_proj_c = nn.Linear(
      config.embedding_size, config.lstm_layers * config.state_size)
    self.p_decoder = LSTMDecoder(config)
    self.p_copy_attn = Attention(
      config.state_size, config.state_size, config.state_size, config.device)
    self.p_copy_g = nn.Linear(config.state_size, 1)
    self.p_switch_g = nn.Linear(config.state_size, 1)
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

  def make_z_potential(self, emission_scores):
    """Prepare the CRF scores/ potentials"""
    batch_size = emission_scores.size(0)
    seq_len = emission_scores.size(1)
    label_size = emission_scores.size(2)

    log_potentials = self.z_crf_transition.view(1, 1, label_size, label_size)\
      .expand(batch_size, seq_len, label_size, label_size) + \
      emission_scores.view(batch_size, seq_len, 1, label_size)\
      .expand(batch_size, seq_len, label_size, label_size)
    return log_potentials

  def encode_z(self, z_sample_ids, z_sample_emb, z_mask):
    """Squeeze the sequence z and encode it
    E.g. z = [1, 1, 2, 2, 2, 3, 3, 4] -> [1, 2, 3, 4]
    z_sample_enc = embeddings([1, 2, 3, 4]).mean(-1)
    
    Args:
      z_sample_ids: size=[batch, max_len]
      z_sample_emb: size=[batch, max_len, emb_size]
      z_mask: size=[batch, max_len]

    Returns:
      z_sample_enc: size=[batch, emb_size]
    """
    # to see if current state is the same as previous state
    batch_size = z_sample_ids.size(0)
    device = z_sample_ids.device
    z_chosen = (z_sample_ids[:, 1:] - z_sample_ids[:, :-1]) != 0
    z_chosen = z_chosen.type(torch.float)
    z_chosen = torch.cat(
      [torch.ones(batch_size, 1).to(device).type(torch.float), z_chosen], 1)
    z_chosen *= z_mask.type(torch.float)
    z_sample_enc = z_sample_emb * z_chosen.unsqueeze(2)
    z_sample_enc = z_sample_enc.sum(dim=1) / z_chosen.sum(dim=1, keepdim=True)
    return z_sample_enc

  def forward(self, keys, vals, sentences, sent_lens, tau, x_lambd):
    """Forward pass, first run the inference network, then run the decoder
    
    Args:
      keys: torch.tensor(torch.long), size=[batch, max_mem_len]
      vals: torch.tensor(torch.long), size=[batch, max_mem_len]
      sentences: torch.tensor(torch.long), size=[batch, sent_len]
      sent_lens: torch.tensor(torch.long), size=[batch]
      tau: gumbel temperature, anneal from 1 to 0.01
      x_lambd: decoder coefficient for the word in, controll how 'autogressive'
       the model is, anneal from 0 to 1 

    Returns:
      loss: torch.float, the total loss 
      out_dict: dict(), output dict  
      out_dict['inspect']: dict(), training process inspection
    """
    out_dict = {}
    inspect = {'latent_state_vocab_ent_avg': None, 
               'latent_state_vocab': None, 
               'latent_state_vocab_ent': None}
    batch_size = sentences.size(0)
    loss = 0.

    ## sentence encoding 
    sent_mask = sentences != self.pad_id
    sentences_emb = self.embeddings(sentences)
    # enc_outputs.shape = [batch, max_len, state_size]
    enc_outputs, (enc_state_h, enc_state_c) =\
      self.q_encoder(sentences_emb, sent_lens)
    # NOTE: max_len != sentences.size(1), max_len = max(sent_lens)
    max_len = enc_outputs.size(1)
    sent_mask = sent_mask[:, : max_len]

    # kv encoding 
    kv_emb, kv_enc, kv_mask = self.encode_kv(keys, vals)

    ## latent template
    # emission score = log potential
    # [batch, max_len, latent_vocab]
    z_emission_scores = self.z_crf_proj(enc_outputs) 
    log_potentials = self.make_z_potential(z_emission_scores)
    chain = torch_struct.LinearChainCRF(log_potentials, sent_lens + 1)

    # entropy regularization
    ent_z = chain.entropy.mean()
    loss += self.z_beta * ent_z
    out_dict['ent_z'] = tmu.to_np(ent_z)
    out_dict['ent_z_loss'] = self.z_beta * tmu.to_np(ent_z)

    # reparameterized sampling
    z_sample = chain.rsample([1], tau)[0].sum(-1)
    _, z_sample_ids = z_sample.max(-1)

    # NOTE: although we use 0 as mask here, 0 is ALSO a valid state 
    z_sample_ids.masked_fill_(~sent_mask, 0) 
    inspect['z_sample_ids'] = z_sample_ids.detach().cpu().numpy()
    z_sample_emb = tmu.seq_gumbel_encode(z_sample, z_sample, z_sample_ids,
      self.z_embeddings, self.gumbel_st)

    # encode z sample 
    # representation of z is contextualized by an LSTM 
    # z_sample_emb, (z_sample_enc, _) = self.z_encoder(z_sample_emb, sent_lens)
    # z_sample_enc = z_sample_enc[-1] # [batch, state]
    z_sample_enc = self.encode_z(z_sample_ids, z_sample_emb, sent_mask)

    # decoding
    sentences = sentences[:, : max_len]
    if(self.dec_adaptive):
      p_log_prob, ppl, switch_g_nll, inspect_ = self.decode_train_adaptive(
        z_sample_ids, z_sample_emb, z_sample_enc, sent_lens,
        vals, kv_emb, kv_enc, kv_mask, sentences, x_lambd)
    else:
      pass # TODO: ADD non-adaptive decoding 
      # p_log_prob, ppl, inspect_ = self.decode_train(
      #   z_sample_ids, z_sample_emb, z_sample_enc, sent_mask,
      #   bow_emb, bow_enc, bow_mask, sentences)
    inspect.update(inspect_)
    out_dict['p_log_prob'] = tmu.to_np(p_log_prob)
    out_dict['neg_ppl'] = tmu.to_np(-ppl)
    out_dict['switch_g_nll'] = tmu.to_np(switch_g_nll)
    loss += p_log_prob
    loss -= switch_g_nll

    # turn maximization to minimization
    loss = -loss

    out_dict['loss'] = tmu.to_np(loss)
    out_dict['inspect'] = inspect
    out_dict.update(inspect)
    return loss, out_dict

  def decode_train_adaptive(self, 
    z_sample_ids, z_sample_emb, z_sample_enc, z_lens,
    mem, mem_emb, mem_enc, mem_mask, sentences, x_lambd):
    """Train the decoder, adaptive version"""
    inspect = {}

    dec_inputs, dec_targets, dec_g =\
      self.prepare_dec_io(
      z_sample_ids, z_sample_emb, z_lens, sentences, x_lambd)
    inspect['dec_targets'] = tmu.to_np(dec_targets)
    inspect['dec_g'] = tmu.to_np(dec_g)

    init_state = self.init_state(mem_enc + z_sample_enc)

    log_prob, ppl, switch_g_nll, inspect_ = self.dec_train_loop(
      init_state, 
      mem, mem_emb, mem_mask, 
      z_sample_enc, z_sample_ids, 
      dec_inputs, dec_targets, dec_g)
    inspect.update(inspect_)
    return log_prob, ppl, switch_g_nll, inspect


  def prepare_dec_io(self, 
    z_sample_ids, z_sample_emb, z_lens, sentences, x_lambd):
    """Prepare the decoder output g based on the inferred z from the CRF 

    E.g.      z = [0,   0,  1,  2,  2]
              x = [GOO, x1, x2, x3, x4], then 
    dec_inputs  = [GOO + 0, x1 + 0, x2 + 1, x3 + 2, x4 + 2]
    dec_outputs = [x1,      x2,     x3,     x4,     EOS]
    dec_g       = [0,       1,      1,      0,      1]

    Args:
      z_sample_ids: size=[batch, max_len]
      z_sample_emb:
      z_lens: size=[batch]
      sentences: size=[batch, max_len]
      x_lambd: 

    Returns:
      dec_inputs: size=[batch, max_len, state]
      dec_targets: size=[batch, max_len]
      dec_g: size=[batch, max_len]
    """
    batch_size = z_sample_ids.size(0)
    max_len = z_sample_ids.size(1)
    dec_g = z_sample_ids[:, :-1] != z_sample_ids[:, 1:]
    dec_g = dec_g.type(torch.long)
    # dec_g = torch.cat([dec_g, torch.ones(batch_size, 1)], dim=1)
    len_one_hot = tmu.ind_to_one_hot(z_lens, max_len - 1)
    dec_g.masked_fill_(len_one_hot.type(torch.bool), 1)

    sent_emb = self.embeddings(sentences)
    dec_inputs = z_sample_emb[:, :-1] + x_lambd * sent_emb[:, :-1]
    dec_targets = sentences[:, 1:]
    return dec_inputs, dec_targets, dec_g


  def dec_train_loop(self, init_state, mem, mem_emb, mem_mask, z_sample_enc, 
    z_sample_ids, dec_inputs, dec_targets, dec_g):
    """Loop for adaptive decoding"""
    inspect = {}

    device = dec_inputs.device
    state_size = self.state_size
    batch_size = dec_inputs.size(0)
    max_len = dec_inputs.size(1)

    dec_cell = self.p_decoder
    state = init_state
    dec_inputs = dec_inputs.transpose(1, 0) # [max_len, batch, state]
    dec_targets = dec_targets.transpose(1, 0)
    dec_g = dec_g.transpose(1, 0)
    log_prob = []
    dec_outputs = []
    switch_g_nll = []
    switch_g_prob = []
    latent_state_vocab = torch.zeros(
      self.latent_vocab_size, self.vocab_size).to(device)
    z_sample_ids = z_sample_ids.transpose(1, 0)

    for i in range(max_len): 
      # word loss 
      dec_out, state = dec_cell(
        dec_inputs[i] + z_sample_enc, state, mem_emb, mem_mask)
      dec_out = dec_out[0]
      lm_logits = dec_cell.output_proj(dec_out)
      lm_prob = F.softmax(lm_logits, dim=-1)
      if(self.use_copy):
        _, copy_dist = self.p_copy_attn(dec_out, mem_emb, mem_mask)
        copy_prob = tmu.batch_index_put(copy_dist, mem, self.vocab_size)
        copy_g = F.sigmoid(self.p_copy_g(dec_out))
        out_prob = (1 - copy_g) * lm_prob + copy_g * copy_prob
        logits = out_prob.log()
        # TODO: memory-efficient loss calculation for pointers
        # log_prob_i = ... 
        log_prob_i = -F.cross_entropy(logits, dec_targets[i], reduction='none')
      else: 
        logits = lm_logits
        out_prob = lm_prob
        log_prob_i = -F.cross_entropy(logits, dec_targets[i], reduction='none')
      
      log_prob.append(log_prob_i) 
      dec_outputs.append(logits.argmax(dim=-1))

      # loss_g
      out_x_prob, out_x = out_prob.max(dim=-1) # [batch]
      out_x_prob = out_x_prob.unsqueeze(1)
      out_emb = self.embeddings(out_x)
      weighted_out_emb =\
        (1-out_x_prob).detach() * out_emb + out_x_prob * out_emb # ST trick
      switch_g_logits = self.p_switch_g(dec_out + weighted_out_emb).squeeze(1)
      switch_g_prob_ = F.sigmoid(switch_g_logits)
      switch_g_nll_ = -dec_g[i] * torch.log(switch_g_prob_ + 1e-10)\
        -(1 - dec_g[i]) * torch.log(1 - switch_g_prob_ + 1e-10)
      switch_g_nll.append(switch_g_nll_)
      switch_g_prob.append(switch_g_prob_)

      # inspection
      latent_state_vocab[z_sample_ids[i]] += out_prob.detach()

    log_prob = torch.stack(log_prob) # [max_len, batch]
    log_prob_ = log_prob.clone()
    mask = dec_targets != self.pad_id # [max_len, batch]
    log_prob.masked_fill_(mask == 0, 0.) 
    log_prob = log_prob.sum() / mask.sum()
    ppl = (-log_prob).detach().exp()

    switch_g_nll = torch.stack(switch_g_nll) # [max_len, batch]
    switch_g_nll.masked_fill_(mask == 0, 0.)
    switch_g_nll = switch_g_nll.sum() / mask.sum()

    switch_g_prob = torch.stack(switch_g_prob)
    switch_g_prob = switch_g_prob.transpose(1, 0)
    inspect['switch_g_prob'] = tmu.to_np(switch_g_prob)

    dec_outputs = torch.stack(dec_outputs).transpose(0, 1)
    inspect['train_predictions'] = tmu.to_np(dec_outputs)
    latent_state_vocab_ent =\
      -latent_state_vocab * torch.log(latent_state_vocab + 1e-10)
    latent_state_vocab_ent = latent_state_vocab_ent.sum(dim=-1)
    inspect['latent_state_vocab_ent'] =\
      latent_state_vocab_ent.detach().cpu().numpy()
    return log_prob, ppl, switch_g_nll, inspect

  def infer(self, keys, vals, z, z_lens, x_lambd):  
    """Latent template inference step

    Args:
      keys: size=[batch, mem_len]
      vals: size=[batch, mem_len]
      z: size=[batch, num_sample, max_len]
      z_lens: size=[batch, num_sample]
    """
    out_dict = {}
    batch_size = keys.size(0)
    num_sample = z.size(1)

    # kv encoding 
    kv_emb, kv_enc, kv_mask = self.encode_kv(keys, vals)

    # z encoding 
    z_ = z.view(batch_size * num_sample, -1)
    z_mask = z_ != self.pad_id
    z_emb = self.z_embeddings(z_)
    z_enc = self.encode_z(z_, z_emb, z_mask)

    # decoding 
    if(self.dec_adaptive):
      predictions, pred_segs = self.decode_infer(
        z_emb, z_enc, vals, kv_emb, kv_enc, kv_mask, x_lambd)
    else: 
      pass # TODO: non-adaptive decoding 

    predictions = tmu.to_np(predictions)
    pred_segs = tmu.to_np(pred_segs)
    predictions = self.post_process(predictions, pred_segs, z, z_lens)

    out_dict['predictions'] = predictions
    out_dict['pred_segs'] = pred_segs
    return out_dict

  def decode_infer(self, 
    z_emb, z_enc, mem, mem_emb, mem_enc, mem_mask, x_lambd):
    """Decode in inference, paralleled version for different samples, greedy 
    decoding 

    Args:
      z: torch.Tensor(), size=[batch, num_sample, max_len]
      z_emb: torch.Tensor(), size=[batch * num_sample, max_len, state_size]
      z_enc: torch.Tensor(), size=[batch * num_sample, state_size]
      mem: torch.Tensor(), size=[batch, mem_len]
      mem_emb: torch.Tensor(), size=[batch, mem_len, state_size]
      mem_enc: torch.Tensor(), size=[batch, state_size]
      mem_mask: torch.Tensor(), size=[batch, mem_len]
      x_lambd: int 
    """
    max_dec_len = self.max_dec_len
    batch_size = mem_enc.size(0)
    num_sample = int(z_enc.size(0) / batch_size)
    mem_len = mem_emb.size(1)
    state_size = mem_emb.size(2)
    device = z_emb.device

    # expand to number of sample 
    mem_enc = mem_enc.view(batch_size, 1, state_size).repeat(1, num_sample, 1)
    mem_enc = mem_enc.view(batch_size * num_sample, state_size)
    mem_emb = mem_emb.view(batch_size, 1, mem_len, state_size)\
      .repeat(1, num_sample, 1, 1)
    mem_emb = mem_emb.view(batch_size * num_sample, mem_len, state_size)
    mem_mask = mem_mask.view(batch_size, 1, mem_len).repeat(1, num_sample, 1)
    mem_mask = mem_mask.view(batch_size * num_sample, mem_len)
    mem = mem.view(batch_size, 1, mem_len).repeat(1, num_sample, 1)
    mem = mem.view(batch_size * num_sample, mem_len)

    # prepare decoding
    state = self.init_state(mem_enc + z_enc)
    predictions = []
    pred_segs = []
    start_tokens = torch.zeros(batch_size * num_sample).to(device)
    start_tokens = start_tokens.type(torch.long) + self.start_id
    prev_x_emb = self.embeddings(start_tokens)
    z_ind = torch.zeros(batch_size * num_sample).to(device).type(torch.long)

    # decoding 
    dec_cell = self.p_decoder
    for i in range(max_dec_len):
      z_emb_ = tmu.batch_index_select(z_emb, z_ind)
      dec_inp = z_emb_ + x_lambd * prev_x_emb + z_enc
      dec_out, state = dec_cell(dec_inp, state, mem_emb, mem_mask)
      dec_out = dec_out[0]

      # lm probability 
      lm_logits = dec_cell.output_proj(dec_out)
      lm_prob = F.softmax(lm_logits, dim=-1)

      # copy probability 
      if(self.use_copy):
        _, copy_dist = self.p_copy_attn(dec_out, mem_emb, mem_mask)
        copy_prob = tmu.batch_index_put(copy_dist, mem, self.vocab_size)
        copy_g = F.sigmoid(self.p_copy_g(dec_out))
        out_prob = (1 - copy_g) * lm_prob + copy_g * copy_prob
        logits = out_prob.log()
      else: 
        out_prob = lm_prob
        logits = lm_logits

      # greedy decoding 
      _, out_x = out_prob.max(dim=-1)
      x_emb = self.embeddings(out_x)
      predictions.append(out_x)

      # update latent state
      switch_g_logits = self.p_switch_g(dec_out + x_emb).squeeze(1)
      switch_g_prob = F.sigmoid(switch_g_logits)
      switch_g = (switch_g_prob > 0.5).type(torch.long)
      pred_segs.append(switch_g)
      z_ind += switch_g

      prev_x_emb = x_emb

    predictions = torch.stack(predictions).transpose(1, 0) # [T, B * N]
    predictions = predictions.view(batch_size, num_sample, max_dec_len)
    pred_segs = torch.stack(pred_segs).transpose(1, 0) # [T, B * N]
    pred_segs = pred_segs.view(batch_size, num_sample, max_dec_len)
    return predictions, pred_segs

  def post_process(self, predictions, pred_segs, z, z_lens):
    """Post processing predictions

    End a sentence if _EOS is predicted or it uses up all the z segments 
    
    Args:
      predictions: size=[batch, num_sample, max_len]
      pred_segs: size=[batch, num_sample, max_len]
      z: size=[batch, num_sample, max_len]
      z_lens: size=[batch, num_sample]
    """

    return predictions