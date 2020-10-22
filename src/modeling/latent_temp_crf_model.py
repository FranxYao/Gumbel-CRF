import torch 
import numpy as np 

from torch import nn 
from torch.optim import Adam, SGD, RMSprop
from torch.nn.utils.clip_grad import clip_grad_norm_

from .latent_temp_crf import LatentTemplateCRF
from .ftmodel import FTModel

class LatentTemplateCRFModel(FTModel):
  def __init__(self, config):
    super(LatentTemplateCRFModel, self).__init__()

    self.model = LatentTemplateCRF(config)
    self.optimizer = Adam(self.model.parameters(), lr=config.learning_rate)
    self.task = config.task 
    self.dataset = config.dataset

    self.z_tau_init = config.z_tau_init
    self.x_lambd_start_epoch = config.x_lambd_start_epoch
    self.x_lambd_increase_interval = config.x_lambd_increase_interval
    self.max_grad_norm = config.max_grad_norm
    self.num_sample_nll = config.num_sample_nll
    self.post_process_start_epoch = config.post_process_start_epoch
    self.dataset = config.dataset
    self.device = config.device
    self.temp_rank_strategy = config.temp_rank_strategy
    return 

  def train_step(self, batch, n_iter, ei, bi, schedule_params):
    # tau annealing
    model = self.model
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
      post_process=post_process,
      debug=debug
      )
    # model.zero_grad()
    loss.backward()
    clip_grad_norm_(model.parameters(), self.max_grad_norm)
    self.optimizer.step()
    out_dict['tau'] = tau
    out_dict['x_lambd'] = x_lambd
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

    if(self.task == 'density' and self.dataset == 'e2e'):
      batch_c, batch = batch
    else:
      batch_c = batch
    # else: raise NotImplementedError(self.task)

    post_process = ei >= self.post_process_start_epoch
    
    with torch.no_grad():
      out_dict = {}

      # likelihood evaluation
      if(self.task == 'density'):
        if(self.dataset == 'e2e'):
          sentences = torch.from_numpy(batch_c['sent_dlex']).to(self.device)
        elif(self.dataset == 'mscoco'):
          sentences = torch.from_numpy(batch_c['sentences']).to(self.device)
        else: 
          raise NotImplementedError('dataset %s not implemented' % self.dataset)
        out_dict_ = model.infer_marginal(
          keys=torch.from_numpy(batch_c['keys']).to(self.device),
          vals=torch.from_numpy(batch_c['vals']).to(self.device),
          sentences=sentences,
          sent_lens=torch.from_numpy(batch_c['sent_lens']).to(self.device),
          tau=tau, 
          x_lambd=x_lambd,
          num_sample=self.num_sample_nll)
        out_dict.update(out_dict_)
      elif(self.task == 'generation'):
        # decode from template
        batch_size = batch['keys'].shape[0]
        
        if(self.temp_rank_strategy == 'random'):
          num_sample = 3
          templates, temp_lens, retrived_keys, retrived_vals, retrived_sents =\
            template_manager.sample(batch_size * num_sample)
        elif(self.temp_rank_strategy == 'closest'):
          num_sample = 1
          templates, temp_lens, retrived_keys, retrived_vals, retrived_sents =\
            template_manager.max_close_to(batch['keys'])
        elif(self.temp_rank_strategy == 'inclusive_closest'):
          num_sample = 1
          templates, temp_lens, retrived_keys, retrived_vals, retrived_sents =\
            template_manager.max_close_inclusive(batch['keys'])
        elif(self.temp_rank_strategy == 'topk'): # set top 3 for now
          num_sample = 3
          close_key_ind = dset.get_close_key_index(bi, mode)
          templates, temp_lens, retrived_keys, retrived_vals, retrived_sents =\
            template_manager.topk_close_to(
              batch['keys'], num_sample, close_key_ind) 
        elif(self.temp_rank_strategy == 'topk-random'): # set top 3 for now
          num_sample = 6
          (templates_0, temp_lens_0, retrived_keys_0, 
            retrived_vals_0, retrived_sents_0) =\
            template_manager.topk_close_to(batch['keys'], 3) 
          (templates_1, temp_lens_1, retrived_keys_1, 
            retrived_vals_1, retrived_sents_1) =\
            template_manager.sample(batch_size * 3)
          templates = np.concatenate(
            [templates_0.reshape(batch_size, 3, -1),
            templates_1.reshape(batch_size, 3, -1)], axis=1)\
            .reshape(batch_size, 6, -1)
          temp_lens = np.concatenate(
            [temp_lens_0.reshape(batch_size, 3),
            temp_lens_1.reshape(batch_size, 3)], axis=1)\
            .reshape(batch_size, 6)
          retrived_keys = np.concatenate(
            [retrived_keys_0.reshape(batch_size, 3, -1),
            retrived_keys_1.reshape(batch_size, 3, -1)], axis=1)\
            .reshape(batch_size, 6, -1)
          retrived_vals = np.concatenate(
            [retrived_vals_0.reshape(batch_size, 3, -1),
            retrived_vals_1.reshape(batch_size, 3, -1)], axis=1)\
            .reshape(batch_size, 6, -1)
          retrived_sents = np.concatenate(
            [retrived_sents_0.reshape(batch_size, 3, -1),
            retrived_sents_1.reshape(batch_size, 3, -1)], axis=1)\
            .reshape(batch_size, 6, -1)
        else: 
          raise NotImplementedError(
            'template sample strategy %s not implemented!' % 
            self.temp_rank_strategy)

        templates = torch.from_numpy(templates).to(self.device).type(torch.long)
        templates = templates.view(batch_size, num_sample, -1)
        temp_lens = torch.from_numpy(temp_lens).to(self.device).type(torch.long)
        temp_lens = temp_lens.view(batch_size, num_sample)

        out_dict['retrived_keys'] = np.reshape(retrived_keys, 
          (batch_size, num_sample, -1))
        out_dict['retrived_vals'] = np.reshape(retrived_vals, 
          (batch_size, num_sample, -1))
        out_dict['retrived_sents'] = np.reshape(retrived_sents, 
          (batch_size, num_sample, -1))
        out_dict['templates'] = tmu.to_np(templates)
        out_dict['temp_lens'] = tmu.to_np(temp_lens)

        out_dict_ = model.infer(
          keys=torch.from_numpy(batch['keys']).to(self.device),
          vals=torch.from_numpy(batch['vals']).to(self.device),
          z=templates, 
          z_lens=temp_lens, 
          x_lambd=x_lambd, 
          post_process=post_process
        )
        out_dict.update(out_dict_)
    return out_dict

  def inspect_grad(self, batch, n_iter, ei, bi, schedule_params):
    # tau annealing
    model = self.model
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

    # gradient check 
    out_dict = model.grad_var(
      keys=torch.from_numpy(batch['keys']).to(self.device),
      vals=torch.from_numpy(batch['vals']).to(self.device),
      sentences=sentences,
      sent_lens=torch.from_numpy(batch['sent_lens']).to(self.device),
      tau=tau, 
      x_lambd=x_lambd,
    )

    print('reparam grad:                 mean: %.4g, std: %.4g, r: %.4g' % 
      (out_dict['g_reparam_mean'], 
      out_dict['g_reparam_std'], 
      out_dict['g_reparam_r']))
    print('score func grad, step level:  mean: %.4g, std: %.4g, r: %.4g' % 
      (out_dict['g_score_step_mean'], out_dict['g_score_step_std'], 
      out_dict['g_score_step_r']))
    print('score func grad, seq level:   mean: %.4g, std: %.4g, r: %.4g' %
      (out_dict['g_score_seq_mean'], 
      out_dict['g_score_seq_std'], 
      out_dict['g_score_seq_r']))

    print('reparam log_p: %.4g' % out_dict['p_log_prob'])
    print('score func step log_p:%.4g, reward: %.4g, learning_sig:%.4g'% 
      (out_dict['log_p_score'], out_dict['reward_step'], 
      out_dict['learning_signal_step']))
    print('score func seq  log_p:%.4g, reward: %.4g, learning_sig:%.4g'% 
      (out_dict['log_p_score'], out_dict['reward_seq'], 
      out_dict['learning_signal_seq']))
    return out_dict