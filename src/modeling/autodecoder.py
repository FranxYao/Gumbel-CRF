
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



class AutoDecoder(nn.Module):
  """The latent template model, CRF version, table to text setting"""

  def __init__(self, config, embeddings=None):
    super(AutoDecoder, self).__init__()
    self.config = config
    self.device = config.device

    self.z_lambd = config.z_lambd
    self.z_beta = config.z_beta
    self.z_gamma = config.z_gamma
    self.z_overlap_logits = config.z_overlap_logits
    self.latent_baseline = config.latent_baseline
    self.num_sample_rl = config.num_sample_rl
    self.gumbel_st = config.gumbel_st
    self.post_process_sampling_enc = config.post_process_sampling_enc
    self.use_copy = config.use_copy
    self.num_sample = config.num_sample

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

  def forward(self, keys, vals, 
    sentences, sent_lens, tau, x_lambd, post_process=False, debug=False):
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

    ## sentence encoding 
    sent_mask = sentences != self.pad_id
    sentences_emb = self.embeddings(sentences)
    # enc_outputs.shape = [batch, max_len, state_size]
    enc_outputs, (enc_state_h, enc_state_c) =\
      self.q_encoder(sentences_emb, sent_lens)
    # NOTE: max_len != sentences.size(1), max_len = max(sent_lens)
    max_len = enc_outputs.size(1)
    sent_mask = sent_mask[:, : max_len]
    sentences = sentences[:, : max_len]

    ## kv encoding 
    kv_emb, kv_enc, kv_mask = self.encode_kv(keys, vals)
    
    ## latent template - change to meaningless template
    if(self.latent_baseline == 'constant'):
      z_sample_ids = torch.zeros_like(sentences) + 1
    elif(self.latent_baseline == 'random'):
      z_sample_ids = torch.randint_like(sentences, 
      low=0, high=self.latent_vocab_size)
    else:
      raise NotImplementedError(
        'latent baseline %s not implemented' % self.latent_baseline)
    z_sample_emb = self.z_embeddings(z_sample_ids)
    z_sample_enc = z_sample_emb.sum(dim=1)

    ## decoding
    p_log_prob, p_log_prob_casewise, inspect_ = self.decode_train(
      z_sample_ids, z_sample_emb, z_sample_enc, sent_lens,
      keys, kv_emb, kv_enc, kv_mask, sentences, x_lambd)

    inspect.update(inspect_)
    out_dict['p_log_prob'] = tmu.to_np(p_log_prob)
    # out_dict['neg_ppl'] = tmu.to_np(-ppl)
    loss += p_log_prob

    ## turn maximization to minimization
    loss = -loss

    out_dict['loss'] = tmu.to_np(loss)
    out_dict['inspect'] = inspect
    out_dict.update(inspect)
    return loss, out_dict

  def decode_train(self, 
    z_sample_ids, z_sample_emb, z_sample_enc, z_lens,
    mem, mem_emb, mem_enc, mem_mask, sentences, x_lambd):
    """Train the decoder, non-adaptive decoding, decoder can be auto-regressive
    or not"""
    inspect = {}

    device = z_sample_ids.device
    state_size = self.state_size
    batch_size = sentences.size(0)

    dec_inputs = z_sample_emb
    dec_targets = sentences

    max_len = dec_inputs.size(1)

    dec_cell = self.p_decoder
    state = self.init_state(mem_enc + z_sample_enc)
    # state = self.init_state(mem_enc)
    dec_inputs = dec_inputs.transpose(1, 0) 
    dec_targets = dec_targets.transpose(1, 0)
    log_prob = []
    dec_outputs = []
    latent_state_vocab = torch.zeros(
      self.latent_vocab_size, self.vocab_size).to(device)
    z_sample_ids = z_sample_ids.transpose(1, 0)

    for i in range(max_len): 
      # word loss 
      dec_out, state = dec_cell(
        dec_inputs[i], state, mem_emb, mem_mask)
      dec_out = dec_out[0]
      lm_logits = dec_cell.output_proj(dec_out)
      lm_prob = F.softmax(lm_logits, dim=-1)
      if(self.use_copy):
        _, copy_dist = self.p_copy_attn(dec_out, mem_emb, mem_mask)
        copy_prob = tmu.batch_index_put(copy_dist, mem, self.vocab_size)
        copy_g = torch.sigmoid(self.p_copy_g(dec_out))
        out_prob = (1 - copy_g) * lm_prob + copy_g * copy_prob
        logits = out_prob.log()
        log_prob_i = -F.cross_entropy(logits, dec_targets[i], reduction='none')
      else: 
        logits = lm_logits
        out_prob = lm_prob
        log_prob_i = -F.cross_entropy(logits, dec_targets[i], reduction='none')
      
      log_prob.append(log_prob_i) 
      dec_outputs.append(logits.argmax(dim=-1))

      latent_state_vocab[z_sample_ids[i]] += out_prob.detach()

    log_prob = torch.stack(log_prob) # [max_len, batch]
    log_prob_ = log_prob.clone()
    mask = dec_targets != self.pad_id # [max_len, batch]
    log_prob.masked_fill_(mask == 0, 0.) 
    log_prob = log_prob.sum() / mask.sum()
    log_prob_casewise = log_prob.sum(dim=0) / mask.sum(dim=0)
    # ppl = (-log_prob).detach().exp()

    dec_outputs = torch.stack(dec_outputs).transpose(0, 1)
    # inspect['train_predictions_stepwise'] = tmu.to_np(dec_outputs)
    latent_state_vocab_ent =\
      -latent_state_vocab * torch.log(latent_state_vocab + 1e-10)
    latent_state_vocab_ent = latent_state_vocab_ent.sum(dim=-1)
    inspect['latent_state_vocab_ent'] =\
      latent_state_vocab_ent.detach().cpu().numpy()
    return log_prob, log_prob_casewise, inspect
