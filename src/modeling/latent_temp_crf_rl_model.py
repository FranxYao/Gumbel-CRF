import torch
import numpy as np 

from torch import nn 
from torch.optim import Adam, SGD, RMSprop
from torch.nn.utils.clip_grad import clip_grad_norm_

from .latent_temp_crf_rl import LatentTemplateCRFRL
from .ftmodel import FTModel

import torch
import traceback
from torch import autograd

class LatentTemplateCRFRLModel(FTModel):
  
  def __init__(self, config):
    super(LatentTemplateCRFRLModel, self).__init__()
    self.model = LatentTemplateCRFRL(config)
    self.optimizer = Adam(self.model.parameters(), lr=config.learning_rate)

    self.z_tau_init = config.z_tau_init
    self.x_lambd_start_epoch = config.x_lambd_start_epoch
    self.x_lambd_increase_interval = config.x_lambd_increase_interval
    self.max_grad_norm = config.max_grad_norm
    self.num_sample_nll = config.num_sample_nll
    self.num_sample_rl = config.num_sample_rl
    self.post_process_start_epoch = config.post_process_start_epoch
    self.dataset = config.dataset
    self.device = config.device
    self.temp_rank_strategy = config.temp_rank_strategy

    self.task = config.task
    return 

  def train_step(self, batch, n_iter, ei, bi, schedule_params, debug=False):
    model = self.model

    # tau annealing
    tau = self.z_tau_init - n_iter * schedule_params['tau_decrease_interval']

    # x_lambd annealing
    if(ei < self.x_lambd_start_epoch):
      x_lambd = 0.
    else:
      x_lambd = ((ei - self.x_lambd_start_epoch) *\
        schedule_params['train_num_batches'] + bi)
      x_lambd *= self.x_lambd_increase_interval 
      if(x_lambd >= 1.): x_lambd = 1.

    post_process = ei >= self.post_process_start_epoch
    if(self.dataset == 'e2e'):
      sentences = torch.from_numpy(batch['sent_dlex']).to(self.device)
    elif(self.dataset == 'mscoco'):
      sentences = torch.from_numpy(batch['sentences']).to(self.device)
    else: 
      raise NotImplementedError('dataset %s not implemented' % self.dataset)

    # DEBUG
    # with GuruMeditation():
    model.zero_grad()
    loss, out_dict = model(
      keys=torch.from_numpy(batch['keys']).to(self.device),
      vals=torch.from_numpy(batch['vals']).to(self.device),
      sentences=sentences,
      sent_lens=torch.from_numpy(batch['sent_lens']).to(self.device),
      tau=tau, 
      x_lambd=x_lambd,
      num_sample=self.num_sample_rl, 
      post_process=post_process,
      debug=debug, 
      return_grad=False
      )
    loss.backward()
    clip_grad_norm_(model.parameters(), self.max_grad_norm)
    self.optimizer.step()

    out_dict['tau'] = tau
    out_dict['x_lambd'] = x_lambd
    return out_dict

  def inspect_grad(self, batch, n_iter, ei, bi, schedule_params):
    model = self.model

    # tau annealing
    tau = self.z_tau_init - n_iter * schedule_params['tau_decrease_interval']

    # x_lambd annealing
    if(ei < self.x_lambd_start_epoch):
      x_lambd = 0.
    else:
      x_lambd = ((ei - self.x_lambd_start_epoch) *\
        schedule_params['train_num_batches'] + bi)
      x_lambd *= self.x_lambd_increase_interval 
      if(x_lambd >= 1.): x_lambd = 1.

    post_process = ei >= self.post_process_start_epoch
    if(self.dataset == 'e2e'):
      sentences = torch.from_numpy(batch['sent_dlex']).to(self.device)
    elif(self.dataset == 'mscoco'):
      sentences = torch.from_numpy(batch['sentences']).to(self.device)
    else: 
      raise NotImplementedError('dataset %s not implemented' % self.dataset)

    debug = False
    
    model.zero_grad()
    loss, out_dict = model(
      keys=torch.from_numpy(batch['keys']).to(self.device),
      vals=torch.from_numpy(batch['vals']).to(self.device),
      sentences=sentences,
      sent_lens=torch.from_numpy(batch['sent_lens']).to(self.device),
      tau=tau, 
      x_lambd=x_lambd,
      num_sample=self.num_sample_rl, 
      post_process=post_process,
      debug=debug, 
      return_grad=True
      )

    retain_keys = [
      'g_seq_mean', 'g_seq_std', 'g_seq_r',
      'g_step_m_mean', 'g_step_m_std', 'g_step_m_r',
      'g_step_bt_mean', 'g_step_bt_std', 'g_step_bt_r',
      'g_step_ut_mean', 'g_step_ut_std', 'g_step_ut_r',
      'log_p_score', 
      'reward_seq', 'learning_signal_seq', 
      'reward_step_m', 'learning_signal_step_m',
      'reward_step_bt', 'learning_signal_step_bt',
      'reward_step_ut', 'learning_signal_step_ut',
      ]
    out_dict_ = {}
    for k in retain_keys: 
      out_dict_[k] = out_dict[k]
    out_dict = out_dict_

    print('score func grad, seq level:   mean: %.4g, std: %.4g, r: %.4g' %
      (out_dict['g_seq_mean'], out_dict['g_seq_std'], out_dict['g_seq_r']))
    print('g_step_m:                     mean: %.4g, std: %.4g, r: %.4g' % 
      (out_dict['g_step_m_mean'], out_dict['g_step_m_std'], 
      out_dict['g_step_m_r']))
    print('g_step_bt:                    mean: %.4g, std: %.4g, r: %.4g' % 
      (out_dict['g_step_bt_mean'], out_dict['g_step_bt_std'], 
      out_dict['g_step_bt_r']))
    print('g_step_ut:                    mean: %.4g, std: %.4g, r: %.4g' % 
      (out_dict['g_step_ut_mean'], out_dict['g_step_ut_std'], 
      out_dict['g_step_ut_r']))

    print('log reconstruction: %.4g' % out_dict['log_p_score'])
    print('score func seq,             reward: %.4g, learning_sig:%.4g'% 
      (out_dict['reward_seq'], out_dict['learning_signal_seq']))
    print('score func step marginal,   reward: %.4g, learning_sig:%.4g'% 
      (out_dict['reward_step_m'], out_dict['learning_signal_step_m']))
    print('score fstep biased trans,   reward: %.4g, learning_sig:%.4g'% 
      (out_dict['reward_step_bt'], out_dict['learning_signal_step_bt']))
    print('score fstep unbiased trans, reward: %.4g, learning_sig:%.4g'% 
      (out_dict['reward_step_ut'], out_dict['learning_signal_step_ut']))
    return out_dict

  def valid_step(self, template_manager, batch, n_iter, ei, bi, 
    mode='dev', dataset=None, schedule_params=None):
    model = self.model 

    # tau annealing
    tau = self.z_tau_init - n_iter * schedule_params['tau_decrease_interval']

    # x_lambd annealing
    if(ei < self.x_lambd_start_epoch):
      x_lambd = 0.
    else:
      x_lambd = ((ei - self.x_lambd_start_epoch) *\
        schedule_params['train_num_batches'] + bi)
      x_lambd *= self.x_lambd_increase_interval 
      if(x_lambd >= 1.): x_lambd = 1.
    
    if(self.dataset == 'e2e'):
      batch_c, batch = batch
    else: batch_c = batch

    post_process = ei >= self.post_process_start_epoch
    if(self.dataset == 'e2e'):
      sentences = torch.from_numpy(batch_c['sent_dlex']).to(self.device)
    elif(self.dataset == 'mscoco'):
      sentences = torch.from_numpy(batch_c['sentences']).to(self.device)
    else: 
      raise NotImplementedError('dataset %s not implemented' % self.dataset)
    
    with torch.no_grad():
      out_dict = {}

      # likelihood evaluation
      if(self.task == 'density'):
        out_dict_ = model.infer_marginal(
          keys=torch.from_numpy(batch_c['keys']).to(self.device),
          vals=torch.from_numpy(batch_c['vals']).to(self.device),
          sentences=sentences,
          sent_lens=torch.from_numpy(batch_c['sent_lens']).to(self.device),
          x_lambd=x_lambd,
          num_sample=self.num_sample_nll)
      elif(self.task == 'generation'):
        # TBC
        pass 
      out_dict.update(out_dict_)

    return out_dict