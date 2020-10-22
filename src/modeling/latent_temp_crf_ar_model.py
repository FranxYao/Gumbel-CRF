import torch 
import numpy as np 

from torch import nn 
from torch.optim import Adam, SGD, RMSprop
from torch.nn.utils.clip_grad import clip_grad_norm_

from .latent_temp_crf_ar import LatentTemplateCRFAR
from .ftmodel import FTModel

class LatentTemplateCRFARModel(FTModel):
  def __init__(self, config):
    super().__init__()

    self.model = LatentTemplateCRFAR(config)
    self.seperate_optimizer = config.seperate_optimizer
    if(self.seperate_optimizer):
      self.enc_optimizer = SGD(
        self.model.inf_parameters(), lr=config.enc_learning_rate)
      self.dec_optimizer = Adam(
        self.model.dec_parameters(), lr=config.dec_learning_rate)
    else:
      self.optimizer = Adam(self.model.parameters(), lr=config.learning_rate)

    self.task = config.task 
    self.dataset = config.dataset

    self.max_grad_norm = config.max_grad_norm
    self.num_sample_nll = config.num_sample_nll
    self.num_sample_rl = config.num_sample_rl
    self.grad_estimator = config.grad_estimator
    self.inspect_grad = config.inspect_grad

    self.dataset = config.dataset
    self.device = config.device
    self.temp_rank_strategy = config.temp_rank_strategy
    return 

  def train_step(self, batch, n_iter, ei, bi, schedule_params):
    model = self.model

    if(self.dataset == 'e2e'):
      sentences = torch.from_numpy(batch['sent_dlex']).to(self.device)
    elif(self.dataset == 'mscoco'):
      sentences = torch.from_numpy(batch['sentences']).to(self.device)
    else: 
      raise NotImplementedError('dataset %s not implemented' % self.dataset)

    model.zero_grad()
    if(self.grad_estimator == 'reparam'):
      loss, out_dict = model(
        keys=torch.from_numpy(batch['keys']).to(self.device),
        vals=torch.from_numpy(batch['vals']).to(self.device),
        sentences=sentences,
        sent_lens=torch.from_numpy(batch['sent_lens']).to(self.device),
        tau=schedule_params['tau'], 
        x_lambd=schedule_params['x_lambd'],
        return_grad=self.inspect_grad
      )
    elif(self.grad_estimator == 'score_func'):
      loss, out_dict = model.forward_score_func(
        keys=torch.from_numpy(batch['keys']).to(self.device),
        vals=torch.from_numpy(batch['vals']).to(self.device),
        sentences=sentences,
        sent_lens=torch.from_numpy(batch['sent_lens']).to(self.device),
        x_lambd=schedule_params['x_lambd'], 
        num_sample=self.num_sample_rl,
        return_grad=self.inspect_grad
      )
    elif(self.grad_estimator == 'rnnlm'):
      loss, out_dict = model.forward_lm(
        sentences=sentences, 
        sent_lens=torch.from_numpy(batch['sent_lens']).to(self.device)
      )
    else: 
      raise NotImplementedError(
        'mode %s not implemented' % self.grad_estimator)

    loss.backward()
    clip_grad_norm_(model.parameters(), self.max_grad_norm)
    self.optimizer.step()

    out_dict['tau'] = schedule_params['tau']
    out_dict['x_lambd'] = schedule_params['x_lambd']
    return out_dict

  def valid_step(self, template_manager, batch, n_iter, ei, bi, 
    mode='dev', dataset=None, schedule_params=None):
    """Single batch validation"""

    model = self.model

    if(self.task == 'density' and self.dataset == 'e2e'):
      batch_c, batch = batch
    else:
      batch_c = batch

    with torch.no_grad():
      out_dict = {}

      if(self.task == 'density'): # likelihood evaluation
        if(self.dataset == 'e2e'):
          sentences = torch.from_numpy(batch_c['sent_dlex']).to(self.device)
        elif(self.dataset == 'mscoco'):
          sentences = torch.from_numpy(batch_c['sentences']).to(self.device)
        else: 
          raise NotImplementedError('dataset %s not implemented' % self.dataset)
        
        if(self.grad_estimator != 'rnnlm'):
          out_dict_ = model.infer_marginal(
            keys=torch.from_numpy(batch_c['keys']).to(self.device),
            vals=torch.from_numpy(batch_c['vals']).to(self.device),
            sentences=sentences,
            sent_lens=torch.from_numpy(batch_c['sent_lens']).to(self.device),
            num_sample=self.num_sample_nll
            )
        else:
          _, out_dict_ =  model.forward_lm(
            sentences=sentences, 
            sent_lens=torch.from_numpy(batch_c['sent_lens']).to(self.device) 
            )
        out_dict.update(out_dict_)
      elif(self.task == 'generation'):
        out_dict = model.infer(
          keys=torch.from_numpy(batch_c['keys']).to(self.device),
          vals=torch.from_numpy(batch_c['vals']).to(self.device)
          )
      else:
        raise NotImplementedError('task %s not implemented!' % self.task)

    return out_dict