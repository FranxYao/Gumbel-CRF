
import copy

import numpy as np 

import torch 
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

from torch.distributions import Categorical

from .lstm_seq2seq.encoder import LSTMEncoder
from .lstm_seq2seq.decoder import LSTMDecoder, Attention
from .structure.linear_crf import LinearChainCRF
from . import torch_model_utils as tmu

class LatentTemplateCRFRL(nn.Module):
  """The latent template model, CRF version, table to text setting"""

  def __init__(self, config, embeddings=None):
    super(LatentTemplateCRFRL, self).__init__()
    self.config = config
    self.device = config.device

    self.z_lambd = config.z_lambd
    self.z_beta = config.z_beta
    self.z_gamma = config.z_gamma
    self.z_b0 = config.z_b0
    self.z_overlap_logits = config.z_overlap_logits
    self.num_sample_rl = config.num_sample_rl
    self.stepwise_reward = config.stepwise_reward
    self.gumbel_st = config.gumbel_st
    self.dec_adaptive = config.dec_adaptive
    self.auto_regressive = config.auto_regressive
    self.post_process_sampling_enc = config.post_process_sampling_enc
    self.use_copy = config.use_copy
    self.use_src_info = config.use_src_info
    self.num_sample = config.num_sample
    self.reward_level = config.reward_level

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
    self.z_crf = LinearChainCRF(config)
    if(config.dec_adaptive == False):
      self.z_encoder = LSTMEncoder(config)

    self.p_dec_init_state_proj_h = nn.Linear(
      config.embedding_size, config.lstm_layers * config.state_size)
    self.p_dec_init_state_proj_c = nn.Linear(
      config.embedding_size, config.lstm_layers * config.state_size)
    self.p_decoder = LSTMDecoder(config)
    self.p_copy_attn = Attention(
      config.state_size, config.state_size, config.state_size)
    self.p_copy_g = nn.Linear(config.state_size, 1)
    self.p_switch_g = nn.Linear(config.state_size, 1)

    config_ = copy.deepcopy(config)
    config_.copy_decoder = True 
    self.post_encoder = LSTMEncoder(config_)
    self.post_decoder = LSTMDecoder(config_)
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

  def encode_z(self, z_emb, z_lens):
    """Treat z as a sequence with fixed length"""
    z_emb, (z_enc, _) = self.z_encoder(z_emb, z_lens)
    return z_emb, z_enc[-1]

  def forward(self, keys, vals, 
    sentences, sent_lens, tau, x_lambd, num_sample, 
    post_process=False, debug=False, return_grad=True):
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
    max_mem_len = keys.size(1)
    loss = 0.
    device = keys.device

    ## sentence encoding 
    sent_mask = sentences != self.pad_id
    sentences_emb = self.embeddings(sentences)
    # enc_outputs.shape = [batch, max_len, state_size]
    enc_outputs, (enc_state_h, enc_state_c) =\
      self.q_encoder(sentences_emb, sent_lens)
    # NOTE: max_len != sentences.size(1), max_len = max(sent_lens)
    max_len = enc_outputs.size(1)
    sent_mask = sent_mask[:, : max_len]

    ## kv encoding 
    kv_emb, kv_enc, kv_mask = self.encode_kv(keys, vals)
    
    ## latent template
    # emission score = log potential
    # [batch, max_len, latent_vocab]
    z_emission_scores = self.z_crf_proj(enc_outputs) 
    if(self.z_overlap_logits):
      z_emission_scores[:, :-1] += z_emission_scores[:, 1:].clone()
      z_emission_scores[:, 1:] += z_emission_scores[:, :-1].clone()

    ## entropy regularization
    ent_z = self.z_crf.entropy(z_emission_scores, sent_lens).mean()
    loss += self.z_beta * ent_z
    out_dict['ent_z'] = tmu.to_np(ent_z)
    out_dict['ent_z_loss'] = self.z_beta * tmu.to_np(ent_z)

    ## hard sampling
    z_sample_ids_all, z_sample_all, z_log_prob_all = [], [], []
    z_marginal_all, z_transition_all = [], []
    for _ in range(num_sample):
      z_sample_ids, z_sample, z_switching, z_log_prob, z_transition =\
        self.z_crf.rsample(
        z_emission_scores, sent_lens, tau, 
        return_switching=True, return_prob=True)
      z_marginal = self.z_crf.marginal(
        z_sample_ids, z_emission_scores, sent_lens)
      z_marginal = tmu.mask_by_length(z_marginal, sent_lens, 0.)

      z_sample_ids.masked_fill_(~sent_mask, 0) 
      z_sample_ids_all.append(z_sample_ids)
      z_sample_all.append(z_sample)
      z_log_prob_all.append(z_log_prob)
      z_transition_all.append(z_transition)
      z_marginal_all.append(z_marginal)

    inspect['z_sample_ids'] = z_sample_ids.detach().cpu().numpy()

    # switching regularization
    loss -= self.z_gamma * z_switching
    inspect['z_switching'] = tmu.to_np(z_switching)
    inspect['z_switching_loss'] = self.z_gamma * tmu.to_np(z_switching)
    # NOTE: although we use 0 as mask here, 0 is ALSO a valid state 

    z_sample_ids = torch.stack(z_sample_ids_all, 1).view(
      batch_size * num_sample, max_len)
    z_sample = torch.stack(z_sample_all, 1).view(
      batch_size * num_sample, max_len, -1)
    z_log_prob = torch.stack(z_log_prob_all, 1).view(
      batch_size * num_sample)
    z_marginal = torch.stack(z_marginal_all, 1) # [B, num_sample, T]
    z_transition = torch.stack(z_transition_all, 1) # [B, num_sample, T]
    
    # [batch_size * num_sample, max_len, state_size]
    z_sample_emb = tmu.seq_gumbel_encode(
      z_sample.detach(), z_sample_ids.detach(),
      self.z_embeddings, True)

    ## decoding
    sentences = sentences[:, : max_len]
    # NOTE: use non-adaptive decoding for now 
    # TODO: an auto-regressive decoder 
    # prepare decoding
    sentences = sentences.view(batch_size, 1, max_len).repeat(
      1, num_sample, 1).view(batch_size * num_sample, -1)
    sent_mask = sent_mask.view(batch_size, 1, max_len).repeat(
      1, num_sample, 1).view(batch_size * num_sample, -1)
    sent_lens = sent_lens.view(batch_size, 1).repeat(1, num_sample).view(
      batch_size * num_sample)
    keys = keys.view(batch_size, 1, -1).repeat(1, num_sample, 1).view(
      batch_size * num_sample, -1)
    kv_enc = kv_enc.view(batch_size, 1, -1).repeat(1, num_sample, 1).view(
      batch_size * num_sample, -1)
    kv_emb = kv_emb.view(batch_size, 1, max_mem_len, -1).repeat(
      1, num_sample, 1, 1).view(batch_size * num_sample, max_mem_len, -1)
    kv_mask = kv_mask.view(batch_size, 1, -1).repeat(1, num_sample, 1).view(
      batch_size * num_sample, -1)

    z_sample_emb, z_sample_enc = self.encode_z(
      z_sample_emb, sent_lens)
    p_log_prob, p_log_prob_casewise, p_log_prob_stepwise, inspect_ =\
      self.decode_train(
      z_sample_ids, z_sample_emb, z_sample_enc, sent_lens,
      keys, kv_emb, kv_enc, kv_mask, sentences, x_lambd)
    # inspect['p_log_prob_stepwise'] = tmu.to_np(p_log_prob_stepwise)

    inspect.update(inspect_)
    out_dict['p_log_prob'] = p_log_prob.item()
    out_dict['log_p_score'] = p_log_prob.item()
    # out_dict['neg_ppl'] = tmu.to_np(-ppl)
    loss += p_log_prob

    elbo = p_log_prob_casewise.view(batch_size, num_sample)
    out_dict['log_prob_sent'] = p_log_prob_casewise.mean().item()
    out_dict['sample_constant'] = np.log(float(num_sample))
    elbo = torch.logsumexp(elbo, 1) - np.log(float(num_sample))
    elbo = elbo.mean()
    out_dict['elbo'] = elbo.item() + ent_z.item()

    ## RL surrogate loss 
    # REINFORCE without control variate
    learning_signal_no_base =\
      (p_log_prob_casewise - self.z_b0).detach() * z_log_prob
    reward_no_base =  p_log_prob_casewise.mean()
    out_dict['reward_no_base'] = (reward_no_base - self.z_b0 + 1e-20).item()
    learning_signal_no_base = self.z_lambd * learning_signal_no_base.mean()
    out_dict['learning_signal_no_base'] = learning_signal_no_base.item()

    # REINFORCE with control variate
    p_log_prob_casewise = p_log_prob_casewise.view(batch_size, num_sample)
    b = p_log_prob_casewise.detach()
    b = (b.sum(dim=1, keepdim=True) - b) / (num_sample - 1)
    # surrogate loss
    z_log_prob = z_log_prob.view(batch_size, num_sample)
    reward_seq = (p_log_prob_casewise - b - self.z_b0).mean().detach()
    out_dict['reward_seq'] = (reward_seq + 1e-20).item()
    learning_signal_seq =\
      (p_log_prob_casewise - b - self.z_b0).detach() * z_log_prob
    # DEBUG
    assert(torch.any(torch.isnan(p_log_prob_casewise)) == False)
    assert(torch.any(torch.isnan(z_log_prob)) == False)
    learning_signal_seq = self.z_lambd * learning_signal_seq.mean() 
    out_dict['learning_signal_seq'] = learning_signal_seq.item()

    # Stepwise reward, marginal version 
    p_log_prob_stepwise = p_log_prob_stepwise.view(
      batch_size, num_sample, -1)
    b = p_log_prob_stepwise.detach()
    b = (b.sum(dim=1, keepdim=True) - b) / (num_sample - 1)
    # NOTE: p_log_prob_stepwise already masked in decoder
    reward_step_m = (p_log_prob_stepwise - b - self.z_b0).mean().detach()
    out_dict['reward_step_m'] = (reward_step_m + 1e-20).item()
    # NOTE: if auto-regressive, z_marginal should be adjusted
    learning_signal_step_m =\
      (p_log_prob_stepwise - b - self.z_b0).detach() * z_marginal
    learning_signal_step_m = self.z_lambd * learning_signal_step_m.mean() 
    out_dict['learning_signal_step_m'] = learning_signal_step_m.item()

    # Stepwise reward, biased transition version 
    learning_signal_step_bt =\
      (p_log_prob_stepwise - b - self.z_b0).detach() * z_transition
    reward_step_bt = (p_log_prob_stepwise - b - self.z_b0).mean().detach()
    out_dict['reward_step_bt'] = (reward_step_bt + 1e-20).item()
    learning_signal_step_bt = self.z_lambd * learning_signal_step_bt.mean()
    out_dict['learning_signal_step_bt'] = learning_signal_step_bt.item()

    # Stepwise reward, unbiased transition version 
    p_log_prob_stepcum = p_log_prob_stepwise.view(batch_size, num_sample, 1, -1)
    p_log_prob_stepcum = p_log_prob_stepcum.repeat(1, 1, max_len, 1)
    cum_mask = torch.triu(torch.ones(max_len, max_len)).to(device)
    sent_mask_ = sent_mask.view(batch_size, num_sample, max_len, 1)
    cum_mask = cum_mask.view(1, 1, max_len, max_len) * sent_mask_.float()
    # print('debug: p_log_prob_stepcum.size =', p_log_prob_stepcum.size())
    # print('debug: cum_mask.size =', cum_mask.size())
    p_log_prob_stepcum = (p_log_prob_stepcum * cum_mask).sum(-1)
    b = p_log_prob_stepcum.detach()
    b = (b.sum(dim=1, keepdim=True) - b) / (num_sample - 1)
    learning_signal_step_ut =\
      (p_log_prob_stepcum - b - self.z_b0).detach() * z_transition
    reward_step_ut = (p_log_prob_stepcum - b - self.z_b0).mean().detach()
    out_dict['reward_step_ut'] = (reward_step_ut + 1e-20).item()
    learning_signal_step_ut = self.z_lambd * learning_signal_step_ut.mean()
    out_dict['learning_signal_step_ut'] = learning_signal_step_ut.item()

    if(self.reward_level == 'seq'):
      learning_signal = learning_signal_seq
    elif(self.reward_level == 'step_marginal'):
      learning_signal = learning_signal_step_m
    elif(self.reward_level == 'step_biased_transition'):
      learning_signal = learning_signal_step_bt
    elif(self.reward_level == 'step_unbiased_transition'):
      learning_signal = learning_signal_step_ut
    # elif(self.reward_level == 'seq_step'):
    #   learning_signal = learning_signal_seq + learning_signal_step
    elif(self.reward_level == 'no_base'):
      learning_signal = learning_signal_no_base
    else: 
      raise NotImplementedError(
        'reward level %s not implemented!' % self.reward_level)

    out_dict['learning_signal'] = tmu.to_np(learning_signal)
    assert(torch.any(torch.isnan(learning_signal)) == False)
    loss += learning_signal

    if(return_grad):
      # sequence level gradient
      self.zero_grad()
      g_seq = torch.autograd.grad(learning_signal_seq, z_emission_scores, 
        retain_graph=True)[0]
      g_seq_mean = g_seq.mean(0)
      g_seq_std = g_seq.std(0)
      g_seq_r = g_seq_std.log() - g_seq_mean.abs().log()
      out_dict['g_seq_mean'] = g_seq_mean.abs().log().mean().item()
      out_dict['g_seq_std'] = g_seq_std.log().mean().item()
      out_dict['g_seq_r'] = g_seq_r.mean().item()

      # stepwise, marginal
      self.zero_grad()
      g_step_m = torch.autograd.grad(learning_signal_step_m, z_emission_scores, 
        retain_graph=True)[0]
      g_step_m_mean = g_step_m.mean(0)
      g_step_m_std = g_step_m.std(0)
      g_step_m_r =\
        g_step_m_std.log() - g_step_m_mean.abs().log()
      out_dict['g_step_m_mean'] =\
        g_step_m_mean.abs().log().mean().item()
      out_dict['g_step_m_std'] = g_step_m_std.log().mean().item()
      out_dict['g_step_m_r'] = g_step_m_r.mean().item()

      # stepwise, biased transitional 
      self.zero_grad()
      g_step_bt = torch.autograd.grad(learning_signal_step_bt, z_emission_scores, 
        retain_graph=True)[0]
      g_step_bt_mean = g_step_bt.mean(0)
      g_step_bt_std = g_step_bt.std(0)
      g_step_bt_r =\
        g_step_bt_std.log() - g_step_bt_mean.abs().log()
      out_dict['g_step_bt_mean'] =\
        g_step_bt_mean.abs().log().mean().item()
      out_dict['g_step_bt_std'] = g_step_bt_std.log().mean().item()
      out_dict['g_step_bt_r'] = g_step_bt_r.mean().item()   

      # stepwise, unbiased transitional 
      self.zero_grad()
      g_step_ut = torch.autograd.grad(learning_signal_step_ut, z_emission_scores, 
        retain_graph=True)[0]
      g_step_ut_mean = g_step_ut.mean(0)
      g_step_ut_std = g_step_ut.std(0)
      g_step_ut_r =\
        g_step_ut_std.log() - g_step_ut_mean.abs().log()
      out_dict['g_step_ut_mean'] =\
        g_step_ut_mean.abs().log().mean().item()
      out_dict['g_step_ut_std'] = g_step_ut_std.log().mean().item()
      out_dict['g_step_ut_r'] = g_step_ut_r.mean().item() 


    ## turn maximization to minimization
    loss = -loss

    out_dict['loss'] = tmu.to_np(loss)
    out_dict['inspect'] = inspect
    out_dict.update(inspect)
    return loss, out_dict

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
      z_sample_emb: size=[batch, max_len, state]
      z_lens: size=[batch]
      sentences: size=[batch, max_len]
      x_lambd: float 

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
    # TODO: word dropout
    dec_inputs = z_sample_emb[:, :-1] + x_lambd * sent_emb[:, :-1]
    dec_targets = sentences[:, 1:]
    return dec_inputs, dec_targets, dec_g

  def decode_train(self, 
    z_sample_ids, z_sample_emb, z_sample_enc, z_lens,
    mem, mem_emb, mem_enc, mem_mask, sentences, x_lambd):
    """Train the decoder, non-adaptive decoding, decoder can be auto-regressive
    or not"""
    inspect = {}

    device = z_sample_ids.device
    state_size = self.state_size
    batch_size = sentences.size(0)

    if(self.auto_regressive):
      dec_inputs, dec_targets, _ = self.prepare_dec_io(
        z_sample_ids, z_sample_emb, z_lens, sentences, x_lambd)
    else: 
      dec_inputs = z_sample_emb
      dec_targets = sentences
    max_len = dec_inputs.size(1)

    dec_cell = self.p_decoder
    if(self.use_src_info):
      state = self.init_state(mem_enc + z_sample_enc)
    else:
      state = self.init_state(z_sample_enc)
    dec_inputs = dec_inputs.transpose(1, 0) 
    dec_targets = dec_targets.transpose(1, 0)
    log_prob = []
    dec_outputs = []
    latent_state_vocab = torch.zeros(
      self.latent_vocab_size, self.vocab_size).to(device)
    z_sample_ids = z_sample_ids.transpose(1, 0)

    for i in range(max_len): 
      # word loss 
      if(self.use_src_info):
        dec_out, state = dec_cell(
          dec_inputs[i], state, mem_emb, mem_mask)
      else:
        dec_out, state = dec_cell(dec_inputs[i], state)
      dec_out = dec_out[0]
      lm_logits = dec_cell.output_proj(dec_out)
      lm_prob = F.softmax(lm_logits, dim=-1)
      if(self.use_copy):
        _, copy_dist = self.p_copy_attn(dec_out, mem_emb, mem_mask)
        copy_prob = tmu.batch_index_put(copy_dist, mem, self.vocab_size)
        copy_g = torch.sigmoid(self.p_copy_g(dec_out))
        out_prob = (1 - copy_g) * lm_prob + copy_g * copy_prob
        logits = (out_prob + 1e-10).log()
        log_prob_i = -F.cross_entropy(logits, dec_targets[i], reduction='none')
      else: 
        logits = lm_logits
        out_prob = lm_prob
        log_prob_i = -F.cross_entropy(logits, dec_targets[i], reduction='none')
      
      log_prob.append(log_prob_i) 
      dec_outputs.append(logits.argmax(dim=-1))

      latent_state_vocab[z_sample_ids[i]] += out_prob.detach()

    log_prob = torch.stack(log_prob) # [max_len, batch]
    mask = dec_targets != self.pad_id # [max_len, batch]
    log_prob.masked_fill_(mask == 0, 0.) 
    log_prob_stepwise = log_prob.transpose(1, 0)
    log_prob_casewise = log_prob.sum(dim=0) # no average over word
    log_prob = log_prob.sum() / mask.sum() # per word log p 
    # ppl = (-log_prob).detach().exp()

    dec_outputs = torch.stack(dec_outputs).transpose(0, 1)
    # inspect['train_predictions_stepwise'] = tmu.to_np(dec_outputs)
    latent_state_vocab_ent =\
      -latent_state_vocab * torch.log(latent_state_vocab + 1e-10)
    latent_state_vocab_ent = latent_state_vocab_ent.sum(dim=-1)
    inspect['latent_state_vocab_ent'] =\
      latent_state_vocab_ent.detach().cpu().numpy()
    return log_prob, log_prob_casewise, log_prob_stepwise, inspect

  def infer(self, keys, vals, z, z_lens, x_lambd, post_process=False):
    """Latent template inference step

    Args:
      keys: size=[batch, mem_len]
      vals: size=[batch, mem_len]
      z: size=[batch, num_sample, max_len]
      z_lens: size=[batch, num_sample]

    Returns:
      out_dict
    """
    out_dict = {}
    batch_size = keys.size(0)
    num_sample = z.size(1)
    max_len = z.size(2)
    mem_len = keys.size(1)
    state_size = self.state_size
    device = keys.device

    # kv encoding 
    kv_emb, kv_enc, kv_mask = self.encode_kv(keys, vals)

    # z encoding 
    z_ = z.view(batch_size * num_sample, -1)
    z_lens = z_lens.view(batch_size * num_sample)
    z_mask = z_ != self.pad_id
    z_emb = self.z_embeddings(z_)
    predictions = torch.zeros(batch_size, num_sample, max_len).type(torch.long)
    pred_segs = torch.zeros(batch_size, num_sample, max_len).type(torch.long)
    predictions = predictions.to(device)
    pred_segs = pred_segs.to(device)

    # decoding 
    if(self.dec_adaptive):
      # z_enc = self.encode_z_adaptive(
      #   z_, z_emb, z_mask)
      z_enc = None
      predictions_, pred_segs_ = self.decode_infer_adaptive(
        z_emb, z_enc, keys, kv_emb, kv_enc, kv_mask, x_lambd)
      max_len_ = predictions_.size(2)
      pred_segs[:, :, :max_len_] = pred_segs_
      pred_segs = tmu.to_np(pred_segs)
      out_dict['pred_segs'] = pred_segs
    else: 
      z_emb, z_enc = self.encode_z(z_emb, z_lens)
      max_len_ = z_emb.size(1)
      predictions_ = self.decode_infer(
        z_emb, z_enc, keys, kv_emb, kv_enc, kv_mask, x_lambd)

    predictions[:, :, :max_len_] = predictions_

    predictions_all = predictions
    predictions = predictions[:, 0, :]
    out_dict['predictions_all'] = tmu.to_np(predictions_all)
    out_dict['predictions'] = tmu.to_np(predictions)
    pred_lens_ = tmu.seq_ends(predictions[:, :-1], self.end_id) + 1
    out_dict['pred_lens'] = tmu.to_np(pred_lens_)

    if(post_process):
      predictions_all = predictions_all.view(batch_size * num_sample, max_len)
      pred_emb = self.embeddings(predictions_all)
      pred_lens = tmu.seq_ends(predictions_all, self.end_id) + 1
      pred_enc, pred_state = self.post_encoder(pred_emb, pred_lens)
      pred_mask = tmu.length_to_mask(pred_lens, pred_enc.size(1))
      kv_emb_ = kv_emb.view(batch_size, 1, mem_len, state_size).repeat(
        1, num_sample, 1, 1).view(batch_size * num_sample, mem_len, state_size)
      kv_mask_ = kv_mask.view(batch_size, 1, mem_len)\
        .repeat(1, num_sample, 1).view(batch_size * num_sample, mem_len)
      keys_ = keys.view(batch_size, 1, mem_len).repeat(1, num_sample, 1)\
        .view(batch_size * num_sample, mem_len)
      post_predictions = self.post_decoder.decode_infer(pred_state, 
        self.embeddings, keys_, kv_emb_, kv_mask_, pred_enc, pred_mask)
      post_predictions = post_predictions.view(batch_size, num_sample, -1)
      out_dict['post_predictions_all'] = tmu.to_np(post_predictions)
      out_dict['post_predictions'] = tmu.to_np(post_predictions[:, 0, :])
      # post_pred_lens = tmu.seq_ends(post_predictions[:, :-1], self.end_id) + 1
      # out_dict['post_pred_lens'] = tmu.to_np(post_pred_lens)

    return out_dict

  def decode_infer(self, z_emb, z_enc, 
    mem, mem_emb, mem_enc, mem_mask, x_lambd, sampling=False, debug=False):
    """Inference, non-adaptive version

    Args:
      z_emb: torch.Tensor(), size=[batch * num_sample, max_len, state_size]
      z_enc: torch.Tensor(), size=[batch * num_sample, state_size]
      mem: torch.Tensor(), size=[batch, mem_len]
      mem_emb: torch.Tensor(), size=[batch, mem_len, state_size]
      mem_enc: torch.Tensor(), size=[batch, state_size]
      mem_mask: torch.Tensor(), size=[batch, mem_len]
    
    Returns:
      predictions: torch.Tensor(int), size=[batch, num_sample, max_dec_len]
    """
    max_dec_len = z_emb.size(1)
    batch_size = mem.size(0)
    num_sample = int(z_emb.size(0) / batch_size)
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

    dec_inputs = z_emb
    dec_inputs = dec_inputs.transpose(1, 0) # [B * N, T, S]
    out_emb = (torch.zeros(batch_size * num_sample) + self.start_id).to(device)
    out_emb = self.embeddings(out_emb.type(torch.long))

    # prepare decoding
    if(z_enc is not None): state = self.init_state(mem_enc + z_enc)
    else: state = self.init_state(mem_enc)
    predictions = []

    # decoding 
    dec_cell = self.p_decoder
    for i in range(max_dec_len):
      if(self.auto_regressive):
        inp = dec_inputs[i] + x_lambd * out_emb
      else: 
        inp = dec_inputs[i]
      if(debug):
        pass
      dec_out, state = dec_cell(inp, state, mem_emb, mem_mask)
      dec_out = dec_out[0]
      lm_logits = dec_cell.output_proj(dec_out)
      lm_prob = F.softmax(lm_logits, dim=-1)
      if(self.use_copy):
        _, copy_dist = self.p_copy_attn(dec_out, mem_emb, mem_mask)
        copy_prob = tmu.batch_index_put(copy_dist, mem, self.vocab_size)
        copy_g = torch.sigmoid(self.p_copy_g(dec_out))
        out_prob = (1 - copy_g) * lm_prob + copy_g * copy_prob
        logits = (out_prob + 1e-10).log()
      else: 
        logits = lm_logits
      if(sampling):
        dist = Categorical(logits=logits)
        out_index = dist.sample()
        predictions.append(out_index)
      else: 
        predictions.append(logits.argmax(dim=-1))
      out_emb = self.embeddings(predictions[-1])

    predictions = torch.stack(predictions).transpose(1, 0)
    predictions = predictions.view(batch_size, num_sample, max_dec_len)
    return predictions

  def infer_marginal(self, keys, vals, 
    sentences, sent_lens, x_lambd, num_sample):
    """Marginal probability and ELBO 
    Via importance sampling from the inference network
    """
    out_dict = {}
    batch_size = sentences.size(0)
    device = sentences.device

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
    if(self.z_overlap_logits):
      z_emission_scores[:, :-1] += z_emission_scores[:, 1:].clone()
      z_emission_scores[:, 1:] += z_emission_scores[:, :-1].clone()

    # entropy regularization
    ent_z = self.z_crf.entropy(z_emission_scores, sent_lens).mean()
    out_dict['ent_z'] = ent_z.item()

    # reparameterized sampling
    z_emission_scores = tmu.batch_repeat(z_emission_scores, num_sample)
    sent_lens_ = sent_lens
    sent_lens = tmu.batch_repeat(sent_lens, num_sample)
    sent_mask = tmu.batch_repeat(sent_mask, num_sample)

    z_sample_ids, z_sample, z_sample_log_prob, _ =\
      self.z_crf.rsample(
      z_emission_scores, sent_lens, tau=1.0, 
      return_switching=False, return_prob=True)
    out_dict['z_sample_log_prob'] = z_sample_log_prob.mean().item()

    # NOTE: although we use 0 as mask here, 0 is ALSO a valid state 
    z_sample_ids.masked_fill_(~sent_mask, 0) 
    # Use hard samples for unbised evaluation
    z_sample_emb = tmu.seq_gumbel_encode(z_sample, z_sample_ids,
      self.z_embeddings, gumbel_st=True)
      
    z_sample_emb, z_sample_enc = self.encode_z(
      z_sample_emb, sent_lens)
 
    sentences = sentences[:, : max_len]
    sentences = tmu.batch_repeat(sentences, num_sample)
    keys = tmu.batch_repeat(keys, num_sample)
    kv_emb = tmu.batch_repeat(kv_emb, num_sample)
    kv_enc = tmu.batch_repeat(kv_enc, num_sample)
    kv_mask = tmu.batch_repeat(kv_mask, num_sample)

    p_log_prob, log_prob_casewise, _, _ = self.decode_train(
      z_sample_ids, z_sample_emb, z_sample_enc, sent_lens,
      keys, kv_emb, kv_enc, kv_mask, sentences, x_lambd)
    
    marginal = log_prob_casewise - z_sample_log_prob
    marginal = marginal.view(batch_size, num_sample)
    marginal = torch.logsumexp(marginal, 1)
    # uniform prior probability 
    prior_log_p = -sent_lens_.float() * np.log(float(self.latent_vocab_size))
    out_dict['prior_log_p'] = prior_log_p.mean().item()
    marginal = marginal - np.log(float(num_sample)) + prior_log_p 
    out_dict['marginal'] = marginal.mean().item()
    out_dict['p_log_prob'] = p_log_prob.item()

    elbo_correct = log_prob_casewise.mean() + ent_z + prior_log_p.mean()
    out_dict['elbo_correct'] = elbo_correct.item() 

    elbo = log_prob_casewise.view(batch_size, num_sample)
    out_dict['log_prob_sent'] = log_prob_casewise.mean().item()
    out_dict['sample_constant'] = np.log(float(num_sample))
    elbo = torch.logsumexp(elbo, 1) - np.log(float(num_sample))
    elbo = elbo.mean()
    out_dict['elbo'] = elbo.item() + ent_z.item()
    return out_dict
