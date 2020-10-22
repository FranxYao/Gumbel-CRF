

import torch
import copy

import numpy as np 
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

from .encoder import LSTMEncoder
from .decoder import LSTMDecoder, Attention
from ..torch_model_utils import bow_to_one_hot, gumbel_softmax_sample

class Seq2seq(nn.Module):
  """The sequence to sequence model"""

  def __init__(self, config, embeddings=None, prior=None):
    """
    Args:
      embeddings: a numpy array, vocab_size * embedding_dim
    """
    super(Seq2seq, self).__init__()
    self.config = config

    self.pad_id = config.pad_id
    self.start_id = config.start_id
    self.end_id = config.end_id
    self.use_struct = config.use_struct
    self.gumbel_temp = config.gumbel_temp

    self.max_dec_len = config.max_dec_len
    self.max_dec_len_test = config.max_dec_len
    self.max_dec_bow = config.max_dec_bow
    self.embedding_size = config.embedding_size
    self.state_size = config.state_size
    self.vocab_size = config.vocab_size
    self.num_bow = config.num_bow

    self.regularize_prior = config.regularize_prior
    self.prior_lambd = config.prior_lambd

    self.use_batch_bow_loss = config.use_batch_bow_loss
    self.batch_bow_lambd = config.batch_bow_lambd
    self.batch_partition = config.batch_partition

    self.use_bow_loss = config.use_bow_loss
    self.bow_lambd = config.bow_lambd

    self.mi_lambd = config.mi_lambd
    self.mi_y_representation = config.mi_y_representation

    self.device = config.device

    self.embeddings = nn.Embedding(config.vocab_size, self.embedding_size)
    if(embeddings is not None): 
      self.embeddings.weight.data.copy_(torch.from_numpy(embeddings))
    if(config.fix_emb): self.embeddings.weight.requires_grad = False 
    if(prior is not None): 
      self.bow_prior = torch.from_numpy(prior).to(self.device)

    bow_vocab_size = config.vocab_size
    self.bow_embeddings = self.embeddings

    self.encoder = LSTMEncoder(config)
    print('bow_vocab_size = %d' % bow_vocab_size)
    self.bow_proj = nn.ModuleList(
      [nn.Linear(2 * self.state_size, bow_vocab_size) 
        for _ in range(config.num_bow)])

    self.decoder = LSTMDecoder(config)
    self.dec_init_state_proj_h = nn.Linear(
      config.embedding_size, config.state_size)
    self.dec_init_state_proj_c = nn.Linear(
      config.embedding_size, config.state_size)

    # conditional mi estimation with jsd representation
    # self.mi_attn = Attention(config.state_size, config.state_size, 
    #   config.embedding_size, config.device)
    # self.mi_proj = nn.Linear(config.state_size, 1)

    # unconditinoal mi estimation with reconstruction and prediction
    # config_ = copy.deepcopy(config)
    # config_.use_struct = False
    # TEST
    # self.mi_dec = LSTMDecoder(config_, mi_dec=False)
    # self.mi_dec = LSTMDecoder(config_, mi_dec=True)

    self.criterion = nn.NLLLoss(reduction='none')

    # self.init_params(if_init_emb=config.use_pretrained_embedding) # TBC 
    return 

  def print(self):
    """Print model trainable parameters"""
    print('\n----------------------------------------------------')
    print('Model parameters:')
    for name, param in self.named_parameters():
      if param.requires_grad:
          print('  ', name, param.data.shape)
    print('----------------------------------------------------\n')
    return 

  def init_params(self, if_init_emb):
    """Initialize the model parameters
    
    Args:
      if_init_emb: if use pretrained embeddings, then do not initialize them 
    """
    # NOTE: cannot do customized initialization now because we have tuned the learning rate before, see if glove is compatible with it first 
    print('Initializing model parameters ... ')

    ## embeddings
    if(if_init_emb == False): pass 
    else: init.normal_(self.embeddings.weight.data)

    ## encoder
    for param in self.encoder.cell.parameters():
      init.normal_(param.data, mean=0, std=0.01)
    init.normal_(self.encoder.bridge_h.weight.data, mean=0, std=0.01)
    init.constant_(self.encoder.bridge_h.bias.data, 0)
    init.normal_(self.encoder.bridge_c.weight.data, mean=0, std=0.01)
    init.constant_(self.encoder.bridge_c.bias.data, 0)
    # initial state is initialized within the encoder 

    ## structure 
    for m in self.bow_proj:
      init.normal_(m.weight.data, mean=0, std=0.01)
      init.constant_(m.bias.data, 0)
    init.normal_(self.dec_init_state_proj_h.weight.data, mean=0, std=0.01)
    init.constant_(self.dec_init_state_proj_h.bias.data, 0)
    init.normal_(self.dec_init_state_proj_c.weight.data, mean=0, std=0.01)
    init.constant_(self.dec_init_state_proj_c.bias.data, 0)

    ## decoder 
    # cell 
    for param in self.decoder.cell.parameters():
      init.normal_(param.data, mean=0, std=0.01)
    # attention
    init.normal_(self.decoder.attention.query_proj.weight.data, 
      mean=0, std=0.01)
    init.constant_(self.decoder.attention.query_proj.bias.data, 0)
    init.normal_(self.decoder.attention.attn_proj.weight.data, 
      mean=0, std=0.01)
    init.constant_(self.decoder.attention.attn_proj.bias.data, 0)
    # bow_attn
    init.normal_(self.decoder.bow_attn.query_proj.weight.data, 
      mean=0, std=0.01)
    init.constant_(self.decoder.bow_attn.query_proj.bias.data, 0)
    init.normal_(self.decoder.bow_attn.attn_proj.weight.data, 
      mean=0, std=0.01)
    init.constant_(self.decoder.bow_attn.attn_proj.bias.data, 0)
    # proj
    init.normal_(self.decoder.attn_cont_proj.weight.data, mean=0, std=0.01)
    init.constant_(self.decoder.attn_cont_proj.bias.data, 0)
    init.normal_(self.decoder.output_proj.weight.data, mean=0, std=0.01)
    init.constant_(self.decoder.output_proj.bias.data, 0)

    print('... finished')
    return 

  def struct_predict(self, enc_out, enc_masks, out_dict, 
    gumbel_sample=False, gumbel_temp=None):
    if(self.use_struct):
      bow_dist = []
      for i in range(self.num_bow):
        logits = self.bow_proj[i](enc_out)
        dist = F.softmax(logits, dim=-1)
        bow_dist.append(dist.unsqueeze(2)) # [B, T, 1, V]
      bow_dist = torch.cat(bow_dist, dim=2) # [B, T, num_bow, V]
      bow_dist = bow_dist.mean(dim=2) # [B, T, V]
      bow_dist_steps = bow_dist
      # print(enc_masks[0])

      # # out_dict['bow_dist_steps'] = bow_dist_steps.detach().cpu().numpy()
      # masks_ = enc_masks.type(torch.float)
      # bow_dist = bow_dist * masks_.unsqueeze(2) # [B, T, 1]
      # # out_dict['bow_dist_steps_masked'] = bow_dist.detach().cpu().numpy()
      # bow_dist = bow_dist.sum(dim=1) / masks_.sum(dim=1, keepdim=True) # [B, V]
      # TEST 
      bow_dist = bow_dist.mean(dim=1)

      if(gumbel_sample):
        if(gumbel_temp is None): gumbel_temp = self.gumbel_temp
        bow_dist = gumbel_softmax_sample(torch.log(bow_dist), self.gumbel_temp)

      out_dict['bow_dist'] = bow_dist.detach().cpu().numpy()
      bow_probs, bow_ind = torch.topk(bow_dist, self.max_dec_bow, dim=-1)
      out_dict['bow_pred_ind'] = bow_ind.detach().cpu().numpy()
      bow_emb = self.bow_embeddings(bow_ind) # [B, sample, S]
      # print('bow_probs.requires_grad:', bow_probs.requires_grad)

      bow_emb = bow_emb * bow_probs.unsqueeze(2)
      # print(bow_emb.shape)
    else:
      bow_emb = torch.zeros(batch_size, self.max_dec_bow, device=device)
      bow_dist = None
    return bow_emb, bow_dist

  def forward(self, 
    enc_inputs, enc_lens, enc_targets, dec_inputs, dec_targets):
    """Forward pass in training
    
    Args:
      enc_inputs: encoder inputs, shape = [batch_size, max_sent_len]
      enc_lens: encoder input length, shape = [batch_size]
      dec_inputs: decoder inputs, shape = [batch_size, max_sent_len]
      dec_targets: decoder outputs, shape = [batch_size, max_sent_len]

    Returns:
      loss: the loss
    """
    out_dict = {}

    # encoder
    enc_mask = enc_inputs != self.pad_id 
    enc_mask = enc_mask[:, : enc_lens.max()]
    enc_inputs = self.embeddings(enc_inputs)
    enc_outputs, enc_state = self.encoder(enc_inputs, enc_lens)

    # sample z 
    if(self.use_struct):
      bow_emb, bow_dist = self.struct_predict(
        enc_outputs, enc_mask, out_dict, 
        gumbel_sample=self.config.train_bow_use_gumbel)
      out_dict['bow_ent'] = torch.sum(
        -bow_dist * torch.log(bow_dist), dim=-1).mean()
      out_dict['bow_ent'] = out_dict['bow_ent'].detach().cpu().numpy()
      # if(self.config.train_dec_init_use_sample):
      #   init_state = (
      #     bow_emb.mean(dim=1).unsqueeze(0), bow_emb.mean(dim=1).unsqueeze(0))
      # else:
      init_state = torch.matmul(bow_dist, self.bow_embeddings.weight)
      init_state = (init_state.unsqueeze(0), init_state.unsqueeze(0))
      init_state = (
        self.dec_init_state_proj_h(init_state[0]) + enc_state[0],
        self.dec_init_state_proj_c(init_state[1]) + enc_state[1])
    else: 
      init_state = enc_state
      bow_emb = None

    loss_lm = self.decode_train(self.decoder, 
                                init_state=init_state, # TBC
                                enc_outputs=enc_outputs, 
                                enc_mask=enc_mask, 
                                bow_emb=bow_emb, 
                                dec_inputs=dec_inputs, 
                                dec_targets=dec_targets)

    loss = loss_lm
    out_dict['loss_lm'] = loss_lm.detach().cpu().numpy()

    if(self.use_struct):
      loss_bow = self.calculate_loss_bow(enc_targets, bow_dist, out_dict)
      loss_bow = self.bow_lambd * loss_bow
      out_dict['loss_bow'] =\
        loss_bow.detach().cpu().numpy() / self.bow_lambd
      if(self.use_bow_loss): loss += loss_bow

      loss_batch_bow = self.calculate_batch_loss_bow_(
        enc_targets, bow_dist, self.batch_partition)
      loss_batch_bow = self.batch_bow_lambd * loss_batch_bow
      out_dict['loss_batch_bow'] =\
        loss_batch_bow.detach().cpu().numpy() / self.batch_bow_lambd
      if(self.use_batch_bow_loss): loss += loss_batch_bow

      # NOTE: use forward KL to encourage mode covering, but ignore the modes 
      # that are not covered 
      aggregate_posterior = bow_dist.mean(dim=0)
      loss_prior = self.prior_lambd * torch.sum(
        aggregate_posterior * torch.log(
          aggregate_posterior / (1e-8 + self.bow_prior)))
      out_dict['loss_prior'] = loss_prior.detach().cpu().numpy()
      if(self.regularize_prior): loss += loss_prior

    out_dict['loss'] = loss.detach().cpu().numpy()
    return loss, out_dict

  def calculate_loss_bow(self, enc_targets, bow_dist, out_dict):
    bow_target_dist = bow_to_one_hot(enc_targets, self.vocab_size, 
      self.pad_id).sum(dim=1)
    bow_target_dist /= bow_target_dist.sum(dim=-1, keepdim=True)

    loss_bow = - bow_target_dist * torch.log(bow_dist) 
    # out_dict['bow_target_dist'] = bow_target_dist.detach().cpu().numpy()
    loss_bow = loss_bow.sum(dim=-1).mean()
    return loss_bow

  def calculate_batch_loss_bow_(self, enc_targets, bow_dist, partition):
    bow_target_dist = bow_to_one_hot(enc_targets, self.vocab_size, 
      self.pad_id).sum(dim=1)
    bow_target_dist /= bow_target_dist.sum(dim=-1, keepdim=True)

    batch_size = int(bow_target_dist.shape[0])
    new_batch_size = batch_size // partition
    assert(batch_size % partition == 0)
    bow_target_dist = bow_target_dist.view(new_batch_size, partition, -1)
    bow_target_dist = bow_target_dist.mean(dim=0)

    bow_dist = bow_dist.view(new_batch_size, partition, -1)
    bow_dist = bow_dist.mean(dim=0)

    loss_batch_bow = - bow_target_dist * torch.log(bow_dist + 1e-10)
    loss_batch_bow = loss_batch_bow.sum(dim=-1).mean()
    return loss_batch_bow

  def decode_train(self, dec_cell, init_state, enc_outputs, enc_mask, bow_emb, 
    dec_inputs, dec_targets, mi_estimate=False):
    """The decoding loop function
  
    Args:
      init_state: the decoder's initial state, a LSTM tupple, 
        each tuple elementshape = [batch_size, state_size]
      enc_outputs: the encoder's outputs states
      enc_mask: the encoder's length mask
      embeddings: the embedding table, shape = [vocab_size, state_size]
      dec_inputs: the decoder input index, shape = [batch_size, max_sent_len]
      dec_targets: the decoder output index, shape = [batch_size, max_sent_len]

    Returns:
      loss: the sequence to sequence loss
    """
    batch_size = dec_inputs.shape[0]
    dec_inputs = dec_inputs.transpose(1, 0) # [T, B]
    # print('dec_inputs.shape', dec_inputs.shape)
    dec_inputs = self.embeddings(dec_inputs) # [T, B, S] 

    state_size = init_state[0][0].shape[1]
    max_dec_len = dec_inputs.shape[0]
    dec_targets = dec_targets.transpose(1, 0) # [T, B]
    mask = dec_targets != self.pad_id

    loss = []
    state = init_state # [2, 1, B, S] 2 = (h, c), 1 = layer size 
    # if(cumu_attn): 
    #   hist_states = torch.zeros(max_dec_len, batch_size, state_size).to(device)
    mi_jsd_1 = []

    # TODO: modify teacher forcing ratio
    for i in range(max_dec_len):
      dec_out, state = dec_cell(
        dec_inputs[i], state, enc_outputs, enc_mask, bow_emb)
      logits = dec_cell.output_proj(dec_out)

      if(mi_estimate):
        if(self.mi_y_representation == 'dec_state'):
          y_rep = dec_out[0]
        # TEST 
        # elif(self.mi_y_representation == 'soft_emb'):
        elif(self.mi_y_representation in ['soft_emb', 'gumbel']):
          dec_dist = F.softmax(logits[0], dim=-1) # [B, V]
          soft_emb = torch.matmul(dec_dist, self.embeddings.weight)
          y_rep = soft_emb
        # elif(self.mi_y_representation == 'gumbel'):
        #   dec_dist = F.softmax(logits[0], dim=-1)
        #   dec_dist = gumbel_softmax_sample(torch.log(dec_dist), self.gumbel_temp)
        #   soft_emb = torch.matmul(dec_dist, self.embeddings.weight)
        #   y_rep = soft_emb
        else: 
          raise NotImplementedError(
            'y representation %s not implemented' % self.mi_y_representation)
        mi_context, _ = self.mi_attn(y_rep, bow_emb)
        mi_jsd_1.append(mi_context.unsqueeze(1)) # [B, 1, S]

      # print(logits.shape)
      # print(dec_targets[i].shape)
      loss_i = self.criterion(F.log_softmax(logits[0], dim=1), dec_targets[i]) # [B]
      loss.append(loss_i)

    loss = torch.stack(loss) # [T, B]
    loss.masked_fill(mask == 0, 0.) 
    loss = loss.sum() / mask.sum()

    if(mi_estimate):
      mi_jsd_1 = torch.stack(mi_jsd_1, dim=1).mean(dim=1)
      mi_jsd_1 = self.mi_proj(mi_jsd_1).squeeze() # [B]
      return loss, mi_jsd_1
    else: 
      return loss

  def infer(self, enc_inputs, enc_lens, test_bow=False):
    """forward pass in evaluation, use greedy decoding for now 
    
    Returns:
      dec_outputs
      dec_log_prob
      dec_log: the log of the intermediate generation process
    """
    out_dict = {}
    with torch.no_grad():
      enc_mask = enc_inputs != self.pad_id 
      enc_inp_index = enc_inputs.detach().cpu().numpy()
      enc_inputs = self.embeddings(enc_inputs)
      enc_outputs, enc_state = self.encoder(enc_inputs, enc_lens)
      enc_mask = enc_mask[:, : enc_lens.max()]
      if(self.use_struct):
        bow_emb, bow_dist = self.struct_predict(enc_outputs, enc_mask, out_dict, 
        gumbel_sample=self.config.infer_bow_use_gumbel)
      else: 
        bow_emb = None
      
      if(test_bow == False):
        if(self.use_struct):
          init_state = torch.matmul(bow_dist, self.bow_embeddings.weight)
          init_state = (init_state.unsqueeze(0), init_state.unsqueeze(0))
          init_state = (
            self.dec_init_state_proj_h(init_state[0]) + enc_state[0],
            self.dec_init_state_proj_c(init_state[1]) + enc_state[1])
        else: init_state = enc_state
        dec_outputs, inspect = self.decode_greedy(dec_cell=self.decoder, 
                                                  init_state=init_state, 
                                                  enc_outputs=enc_outputs, 
                                                  enc_mask=enc_mask, 
                                                  bow_emb=bow_emb)
      else:
        if(self.config.infer_dec_init_use_sample):
          init_state = (bow_emb.mean(dim=1).unsqueeze(0), 
            bow_emb.mean(dim=1).unsqueeze(0))
        else:
          init_state = torch.matmul(bow_dist, self.bow_embeddings.weight)
          init_state = (init_state.unsqueeze(0), init_state.unsqueeze(0))
        init_state = (
          self.dec_init_state_proj_h(init_state[0]),
          self.dec_init_state_proj_c(init_state[1]))
        dec_outputs, inspect = self.decode_greedy(
          self.mi_dec, 
          init_state, bow_emb, enc_mask=None, bow_emb=None)
    out_dict['dec_predict'] = dec_outputs.detach().cpu().numpy()
    return out_dict

  def decode_greedy(self, dec_cell, 
    init_state, enc_outputs, enc_mask, bow_emb, mi_estimate=False):
    dec_proj = dec_cell.output_proj
    dec_outputs = []
    batch_size = enc_outputs.shape[0]
    dec_log = None # decoding middle stages, TBC 
    embeddings = self.embeddings
    device = self.device
    dec_start_id = self.start_id

    inp = embeddings(
      torch.zeros(batch_size, dtype=torch.long).to(device) + dec_start_id) 
    state_size = init_state[0][0].shape[1]
    state = init_state
    dec_log_prob = []
    dec_logits = []
    mi_jsd_0 = []
    inspect = {}
    
    for i in range(self.max_dec_len):
      out, state = dec_cell(inp, state, enc_outputs, enc_mask, bow_emb)
      logits = dec_proj(out)

      if(mi_estimate):
        if(self.mi_y_representation == 'dec_state'):
          y_rep = out[0]
        elif(self.mi_y_representation == 'soft_emb'):
          dec_dist = F.softmax(logits[0], dim=-1) # [B, V]
          soft_emb = torch.matmul(dec_dist, self.embeddings.weight)
          y_rep = soft_emb
        elif(self.mi_y_representation == 'gumbel'):
          dec_dist = F.softmax(logits[0], dim=-1)
          dec_dist = gumbel_softmax_sample(torch.log(dec_dist), self.gumbel_temp)
          soft_emb = torch.matmul(dec_dist, self.embeddings.weight)
          y_rep = soft_emb
        else: 
          raise NotImplementedError(
            'y representation %s not implemented' % self.mi_y_representation)

        mi_context, _ = self.mi_attn(y_rep, bow_emb)
        mi_jsd_0.append(mi_context.unsqueeze(1)) # [B, 1, S]
  
      log_prob_i, out_index = torch.max(logits[0], 1)
      dec_logits.append(logits[0])
      dec_log_prob.append(log_prob_i)
      dec_outputs.append(out_index)
      inp = embeddings(out_index)

    dec_outputs = torch.stack(dec_outputs).transpose(1, 0) # [B, T]
    # print('dec_outputs.shape', dec_outputs.shape)
    dec_logits = torch.stack(dec_logits).transpose(1, 0) # [B, T, V]

    if(mi_estimate): 
      mi_jsd_0 = torch.stack(mi_jsd_0, dim=1).mean(dim=1)
      mi_jsd_0 = self.mi_proj(mi_jsd_0).squeeze() # [B]
      inspect['mi_jsd_0'] = mi_jsd_0.mean().detach().cpu().numpy()
      return dec_outputs, mi_jsd_0, inspect
    else: 
      return dec_outputs, inspect

  def forward_test_bow(self, 
    enc_inputs, enc_lens, enc_targets, dec_inputs, dec_targets):
    """Forward pass, use only BOW as the code """
    out_dict = {}

    # encoder 
    enc_mask = enc_inputs != self.pad_id 
    enc_mask = enc_mask[:, : enc_lens.max()]
    enc_inputs_emb = self.embeddings(enc_inputs)
    enc_outputs, enc_state = self.encoder(enc_inputs_emb, enc_lens)

    # sample z 
    loss_y = 0
    for _ in range(self.config.bow_sample_size):
      bow_emb, bow_dist = self.struct_predict(
        enc_outputs, enc_mask, out_dict, 
        gumbel_sample=self.config.train_bow_use_gumbel)
      out_dict['bow_ent'] = torch.sum(
        -bow_dist * torch.log(bow_dist), dim=-1).mean()
      out_dict['bow_ent'] = out_dict['bow_ent'].detach().cpu().numpy()

      if(self.config.train_dec_init_use_sample):
        init_state = (
          bow_emb.mean(dim=1).unsqueeze(0), bow_emb.mean(dim=1).unsqueeze(0))
      else:
        init_state = torch.matmul(bow_dist, self.bow_embeddings.weight)
        init_state = (init_state.unsqueeze(0), init_state.unsqueeze(0))
      init_state = (
        self.dec_init_state_proj_h(init_state[0]),
        self.dec_init_state_proj_c(init_state[1]))

      loss_y_ = self.decode_train(self.mi_dec, 
        init_state=init_state, enc_outputs=bow_emb, enc_mask=None, 
        bow_emb=None, 
        dec_inputs=dec_inputs, dec_targets=dec_targets)
      # loss_y = self.decode_train(self.mi_dec, 
      #   init_state=enc_state, enc_outputs=enc_outputs, enc_mask=enc_mask, 
      #   bow_emb=None, 
      #   dec_inputs=dec_inputs, dec_targets=dec_targets)
      loss_y += loss_y_

    loss_y /= self.config.bow_sample_size
    out_dict['mi_y_reconstruct'] = loss_y.detach().cpu().numpy()
    loss = loss_y

    loss_bow = self.calculate_loss_bow(enc_targets, bow_dist, out_dict)
    loss_bow = self.bow_lambd * loss_bow
    out_dict['loss_bow'] =\
      loss_bow.detach().cpu().numpy() / self.bow_lambd
    if(self.use_bow_loss): loss += loss_bow

    # loss_batch_bow = self.calculate_batch_loss_bow(enc_targets, bow_dist)
    loss_batch_bow = self.calculate_batch_loss_bow_(
      enc_targets, bow_dist, self.batch_partition)
    loss_batch_bow = self.batch_bow_lambd * loss_batch_bow
    out_dict['loss_batch_bow'] =\
      loss_batch_bow.detach().cpu().numpy() / self.batch_bow_lambd
    if(self.use_batch_bow_loss): loss += loss_batch_bow

    aggregate_posterior = bow_dist.mean(dim=0)
    # NOTE: use forward KL to encourage mode covering, but ignore the modes 
    # that are not covered 
    loss_prior = self.prior_lambd * torch.sum(
      aggregate_posterior * torch.log(
        aggregate_posterior / (1e-8 + self.bow_prior)))
    out_dict['loss_prior'] = loss_prior.detach().cpu().numpy()
    if(self.regularize_prior): loss += loss_prior

    out_dict['loss'] = loss.detach().cpu().numpy()
    out_dict['loss_lm'] = loss_y.detach().cpu().numpy()
    return loss, out_dict

  def report_grad(self):
    """Report the average norm the gradient, used for monitoring training"""
    print('gradient of the model parameters:')

    grad_norms = {}
    grad_std = {}
    for name, param in self.named_parameters():
      frist_level_name = name.split('.')[0]
      if(param.requires_grad and param.grad is not None):
        # print(name, param.grad.norm(), param.grad.std())
        if(frist_level_name in grad_norms):
          grad_norms[frist_level_name].append(
            param.grad.norm().detach().cpu().numpy())
          grad_std[frist_level_name].append(
            param.grad.std().detach().cpu().numpy())
        else: 
          grad_norms[frist_level_name] = [
            param.grad.norm().detach().cpu().numpy()]
          grad_std[frist_level_name] = [
            param.grad.std().detach().cpu().numpy()]
    
    for frist_level_name in grad_norms:
      print(frist_level_name, 
        np.average(grad_norms[frist_level_name]),
        np.std(grad_std[frist_level_name]))
    return 
  


