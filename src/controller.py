import torch 
import sys 
import copy

import numpy as np 

from modeling import torch_model_utils as tmu
from modeling.rnnlm import RNNLM

from torch import nn 
from torch.optim import Adam, SGD, RMSprop
from torch.nn.utils.clip_grad import clip_grad_norm_

from time import time
from tqdm import tqdm
from pprint import pprint 

import rouge
from nltk.translate.bleu_score import corpus_bleu

from logger import TrainingLog 
from template_manager import TemplateManager
from tensorboardX import SummaryWriter

class Controller(object):
  """Controller for training, validation, and evaluation

  Controller contains: 
    * logger
    * tensorboard_writer
    * optimizer
    * evaluator
  And it does:
    * train(): loop over the training set, get a batch of data, train the model, 
      log the loss and other metrics. 
      After each epoch it calls the validate() function, check if the model has
      improved, and store the model if it is. 
    * validate(): loop over the dev set, get a batch of data, get the 
      predictions, log the metrics. 
  """

  def __init__(self, config, model, dataset):
    super(Controller, self).__init__()

    self.model_name = config.model_name
    self.model_version = config.model_version
    self.inspect_model = config.inspect_model
    self.inspect_grad = config.inspect_grad
    self.dataset = config.dataset
    self.task = config.task

    self.is_test = config.is_test

    validation_scores = config.validation_scores
    self.validation_scores = {}
    for n in validation_scores:
      self.validation_scores[n] = []

    self.num_epoch = config.num_epoch
    self.start_epoch = config.start_epoch 
    self.validate_start_epoch = config.validate_start_epoch
    self.print_interval = config.print_interval
    self.model_path = config.model_path
    self.output_path = config.output_path
    self.device = config.device
    self.batch_size_train = config.batch_size_train
    self.batch_size_eval = config.batch_size_eval
    self.end_id = config.end_id
    self.latent_vocab_size = config.latent_vocab_size

    self.use_gumbel = config.use_gumbel

    self.z_tau_init = config.z_tau_init
    self.z_tau_final = config.z_tau_final
    self.tau_anneal_epoch = config.tau_anneal_epoch
    self.x_lambd_start_epoch = config.x_lambd_start_epoch
    self.x_lambd_anneal_epoch = config.x_lambd_anneal_epoch
    self.num_sample = config.num_sample
    self.num_sample_rl = config.num_sample_rl
    self.num_sample_nll = config.num_sample_nll
    self.max_dec_len = config.max_dec_len

    self.num_sample_nll = config.num_sample_nll

    self.temp_rank_strategy = config.temp_rank_strategy
    self.decode_strategy = config.decode_strategy

    self.max_grad_norm = config.max_grad_norm
    self.p_max_grad_norm = config.p_max_grad_norm
    self.q_max_grad_norm = config.q_max_grad_norm

    self.validation_criteria = config.validation_criteria
    self.test_validate = config.test_validate

    self.config = config
    self.use_tensorboard = config.use_tensorboard
    self.write_full = config.write_full_predictions
    self.save_ckpt = config.save_ckpt

    self.schedule_params = dict()

    # template manager
    if(self.model_name.startswith('latent_temp') and config.save_temp):
      self.template_manager = TemplateManager(config, dataset.id2word)
    else: self.template_manager = None

    # logging 
    self.logger = TrainingLog(config) 
    if(self.use_tensorboard and (self.is_test == False)):
      print('writting tensorboard at:\n  %s' % config.tensorboard_path)
      self.tensorboard_writer = SummaryWriter(config.tensorboard_path)
    else: 
      self.tensorboard_writer = None

    # evaluate
    self.evaluator = rouge.Rouge(
      metrics=['rouge-n', 'rouge-l'],
      max_n=2,
      limit_length=True,
      length_limit=100,
      length_limit_type='words',
      apply_avg=True,
      apply_best=False,
      alpha=0.5, # Default F1_score
      weight_factor=1.2,
      stemming=True)

    # if('ppl' in self.validation_scores):
    #   print('Loading rmmln .. ')
    #   self.lm = RNNLM(config)
    #   checkpoint = torch.load(config.lm_pretrained_path)
    #   self.lm.load_state_dict(checkpoint['model_state_dict'])
    #   self.lm.to(config.device)
    #   self.lm.eval()
    return 

  def save(self, model, ei):
    """Save the model after epoch"""
    # TODO: different save function for different models 
    # (if using seperate optimizers)
    save_path = self.model_path + 'ckpt_e%d' % ei
    print('Saving the model at: %s' % save_path)
    torch.save(
      {'model_state_dict': model.state_dict(), 
        'optimizer_state_dict': model.optimizer.state_dict()}, 
      save_path)
    return 

  def write_tensorboard(self, out_dict, n_iter, mode, key=None):
    if(key is None):
      for metrics in out_dict:
        if(metrics in self.logger.log):
          self.tensorboard_writer.add_scalar('%s/' % mode + metrics, 
            out_dict[metrics], n_iter)
    else:
      self.tensorboard_writer.add_scalar('%s/' % mode + key, 
        out_dict[key], n_iter)
    return

  def train_schedule_init(self, num_batches, start_epoch, num_epoch):
    """Training schedule initialization"""
    tau_total_iter = num_batches * self.tau_anneal_epoch
    self.train_num_batches = num_batches
    self.schedule_params['tau_decrease_interval'] =\
      (self.z_tau_init - self.z_tau_final) / (tau_total_iter - 1)

    if(self.x_lambd_anneal_epoch > 0):
      tau_total_iter_lambd = num_batches * self.x_lambd_anneal_epoch
      self.schedule_params['x_lambd_decrease_interval'] =\
        1 / (tau_total_iter_lambd - 1)
    else: 
      self.schedule_params['x_lambd_decrease_interval'] = 1

    self.schedule_params['tau'] = self.z_tau_init
    self.schedule_params['x_lambd'] = 1.
    return 

  def scheduler_step(self, n_iter, ei, bi):
    self.schedule_params['tau'] -= self.schedule_params['tau_decrease_interval']
    if(self.schedule_params['tau'] < self.z_tau_final): 
      self.schedule_params['tau'] = self.z_tau_final

    if(ei >= self.x_lambd_start_epoch):
      self.schedule_params['x_lambd'] -=\
        self.schedule_params['x_lambd_decrease_interval']
      if(self.schedule_params['x_lambd'] < 0.): 
        self.schedule_params['x_lambd'] = 0.
    return 

  def train(self, model, dataset):
    """Train the model"""
    print('Start training ... ')

    best_validation = -1e10
    best_validation_epoch = -1
    best_validation_scores = None
    start_time = time()

    num_batches = dataset.num_batches('train', self.batch_size_train)
    print('train dataset, %d batches in total' % num_batches)

    self.train_schedule_init(num_batches, self.start_epoch, self.num_epoch)

    n_iter = self.start_epoch * num_batches - 1
    for ei in range(self.start_epoch, self.num_epoch):
      model.train()
      # before epoch 
      self.logger.reset()
      if(self.template_manager is not None): self.template_manager.clear()
      epoch_start_time = time()

      for bi in range(num_batches):
        n_iter += 1
        batch = dataset.next_batch('train', self.batch_size_train)
        self.scheduler_step(n_iter, ei, bi)
        out_dict = model.train_step(batch, n_iter, ei, bi, self.schedule_params)
        self.logger.update(out_dict)
        if(self.use_tensorboard):
          self.write_tensorboard(out_dict, n_iter, 'train')
        if(self.template_manager is not None):
          self.template_manager.add_batch(batch, out_dict)

        if(bi % self.print_interval == 0): 

          print(
            '\nmodel %s %s epoch %d/%d batch %d/%d n_iter %d time %ds time per batch %.2fs' % 
            (self.model_name, self.model_version, 
              ei, self.num_epoch, bi, num_batches, n_iter, time() - start_time, 
              (time() - epoch_start_time) / (bi + 1)))
          # print the average metrics starting from current epoch 
          self.logger.print()
          tmu.print_grad(model) 

          if(self.inspect_model):
            dataset.print_inspect(out_dict['inspect'], batch, self.model_name)

          # if(self.inspect_grad):
          #   out_dict = model.inspect_grad(batch, n_iter, ei, bi, self.schedule_params)
          #   self.logger.update(out_dict)

        if(bi % (self.print_interval // 5) == 0):
          print('.', end=' ', flush=True)
        
        # start with a validation
        if(bi == 2 and ei == 0 and self.test_validate): 
          _, scores = self.validate(
            model, dataset, -1, n_iter, 'dev', self.tensorboard_writer) 
          pprint(scores)

      # after epoch 
      print('model %s %s epoch %d finished, time: %d' % 
        (self.model_name, self.model_version, ei, time() - start_time))
      self.logger.print() 
      
      print('----------------------------------------------------------------')
      if(ei >= self.validate_start_epoch):
        if(self.template_manager is not None and self.task == 'generation'):
          self.template_manager.report_statistics(dataset.id2word, ei)
          self.template_manager.save(ei)

        validation_criteria, validation_scores = self.validate(
          model, dataset, ei, n_iter, 'dev', self.tensorboard_writer)

        if(validation_criteria > best_validation):
          print(
            'validation increase from %.4f to %.4f, save the model' %
            (best_validation, validation_criteria))
          print('current validation score:')
          pprint(validation_scores)
          best_validation = validation_criteria
          best_validation_epoch = ei
          best_validation_scores = copy.deepcopy(validation_scores)
          # save model 
          if(self.save_ckpt):
            self.save(model, ei)
        else: 
          print(
            'Validation %.4f, no improvement, keep best at epoch %d' % 
            (validation_criteria, best_validation_epoch))
          print('current validation score:')
          pprint(validation_scores)
          print('best validation score:')
          pprint(best_validation_scores)
        print('----------------------------------------------------------------')
        print()
        _, test_scores = self.validate(
          model, dataset, ei, n_iter, 'test', self.tensorboard_writer)
        print('test scores:')
        pprint(test_scores)
      else: 
        print('validate_start_epoch = %d, current %d, do not validate' % 
          (self.validate_start_epoch, ei))
    
    self.validate(model, dataset, -1, -1, 'test')
    return

  def validate(self, model, dataset, ei, n_iter, 
    mode='dev', tensorboard_writer=None):
    """
    Args:
      mode: 'dev' or 'test'
    """
    # predictions visualization TBC 
    print('Model %s_%s, epoch %d, n_iter %d, validation on %s set ..' % 
      (self.model_name, self.model_version, ei, n_iter, mode))
    model.eval()

    fd = open(self.output_path +\
      self.model_name + '_' + mode + '_epoch_%d.txt' % ei, 'w')
    if(self.write_full):
      fd_full = open(self.output_path +\
        self.model_name + '_' + mode + '_epoch_%d_full.txt' % ei, 'w')
    else: fd_full = None

    num_batches = dataset.num_batches(mode, self.batch_size_eval)

    print('%d batches in total' % num_batches)
    print_interval = 50
    inspect_at = np.random.randint(print_interval)
    start_time = time()
    hyps = None
    post_hyps = None
    hyps_self = None
    refs = []
    refs_self = None
    scores = copy.deepcopy(self.validation_scores)

    for bi in range(num_batches):
      batch = dataset.next_batch(mode, self.batch_size_eval)
      out_dict = model.valid_step(self.template_manager, batch, n_iter, ei, bi,
        mode, dataset, self.schedule_params)

      if(self.task == 'density' and self.model_name == 'latent_temp_crf' and
        self.dataset == 'e2e'):
        batch = batch[1]
      if(self.dataset == 'e2e'):
        dataset.post_process(batch, out_dict)

      for n in out_dict:
        if(n in scores): scores[n].append(out_dict[n])

      if('predictions' in out_dict):
        if(hyps is None): hyps = out_dict['predictions']
        else: hyps = np.concatenate([hyps, out_dict['predictions']], axis=0)

      if('references' in batch):
        refs.extend(batch['references'])

      # if('predictions_all' in out_dict):
      #   if( (self.model_name == 'latent_temp_crf' and 
      #       self.temp_rank_strategy in ['random', 'topk'])
      #       or 
      #       self.model_name in ['gaussian_vae', 'seq2seq', 'kv2seq']):
      #     if(hyps_self is None): hyps_self = out_dict['predictions_all'][:, 0]
      #     else: 
      #       hyps_self = np.concatenate(
      #         [hyps_self, out_dict['predictions_all'][:, 0]], axis=0)
      #     refs_self.extend(out_dict['predictions_all'][:, 1:])
      
      if('sentences' in batch and self.dataset == 'mscoco'):
        if(refs_self is None): 
          refs_self = batch['sentences']
        else: 
          refs_self = np.concatenate([refs_self, batch['sentences']], axis=0)

      if(bi % 20 == 0): 
        print('.', end=' ', flush=True)

      if('predictions' in out_dict):
        dataset.print_batch(
          batch, out_dict, self.model_name, fd, fd_full)
      
    fd.close()
    if(self.write_full): fd_full.close()

    for n in scores: 
      if(len(scores[n]) != 0): scores[n] = float(np.average(scores[n]))
      else: scores[n] = -1

    if('predictions' in out_dict):
      scores_ = self.eval_scores(hyps, refs, dataset)
      scores.update(scores_)

    if(refs_self is not None and self.dataset == 'mscoco'):
      refs_self = [[si] for si in refs_self]
      scores_ = self.eval_scores(hyps, refs_self, dataset)
      sk = list(scores_.keys())
      for k in sk:
        scores_['self_' + k] = scores_[k]
        scores_['i' + k] = 0.9 * scores[k] - 0.1 * scores_['self_' + k]
        del scores_[k]
      scores.update(scores_)


    if(tensorboard_writer is not None):
      for n in scores:
        if(isinstance(scores[n], float)):
          tensorboard_writer.add_scalar(mode + '/' + n, scores[n], n_iter)

      
    print('')
    if('predictions' in out_dict and mode == 'dev'):
      dataset.print_batch(batch, out_dict, self.model_name)
    print('validation finished, time: %.2f' % (time() - start_time))

    scores['epoch'] = ei 

    model.train()

    return scores[self.validation_criteria], scores

  def eval_ppl(self, out_dict):
    """Evaluate the perplexity of generated sentences"""
    ret = {}
    with torch.no_grad():
      _, lm_out_dict = self.lm(
        torch.from_numpy(out_dict['predictions'][:, :-1]).to(self.device),
        torch.from_numpy(out_dict['predictions'][:, 1:]).to(self.device),
        torch.from_numpy(out_dict['pred_lens']).to(self.device))
      ret['ppl'] = -lm_out_dict['neg_ppl']

      if('post_predictions' in out_dict):
        print('with post predictions!')
        _, lm_out_dict = self.lm(
          torch.from_numpy(
            out_dict['post_predictions'][:, :-1]).to(self.device),
          torch.from_numpy(
            out_dict['post_predictions'][:, 1:]).to(self.device),
          torch.from_numpy(out_dict['post_pred_lens']).to(self.device))
        ret['post_ppl'] = -lm_out_dict['neg_ppl']
    return ret
  
  def eval_scores(self, hyps, refs, dataset):
    """
    Args:
      hyps: a list of sentences, each sentence is a list of index 
      refs: a list of reference sets, each reference set is a list of sentences,
        each sentence is a list of index  
    """
    ## rouge score 
    scores = {}
    def _cut_eos(s, end_id):
      s_ = []
      for w in s:
        if(w == end_id): break
        s_.append(w)
      return s_

    refs_ = []
    for r in refs:
      r_ = []
      for ri in r:
        r_.append(dataset.decode_sent(ri))
      refs_.append(r_)
    hyps_ = [dataset.decode_sent(_cut_eos(s, self.end_id)) for s in hyps]
    rouge_scores = self.evaluator.get_scores(hyps_, refs_)
    # pprint(rouge_scores)
    scores['r1'] = float(rouge_scores['rouge-1']['r'])
    scores['r2'] = float(rouge_scores['rouge-2']['r'])
    scores['rl'] = float(rouge_scores['rouge-l']['r'])

    ## bleu score 
    hyps_ = [_cut_eos(s, self.end_id) for s in hyps]
    refs_ = []
    for r in refs:
      r_ = [_cut_eos(ri, self.end_id) for ri in r]
      refs_.append(r_)

    bleu_scores = {}
    bleu_scores['bleu_1'] = corpus_bleu(
      refs_, hyps_, weights=(1., 0, 0, 0))
    bleu_scores['bleu_2'] = corpus_bleu(
      refs_, hyps_, weights=(0.5, 0.5, 0, 0))
    bleu_scores['bleu_3'] = corpus_bleu(
      refs_, hyps_, weights=(0.333, 0.333, 0.333, 0))
    bleu_scores['bleu_4'] = corpus_bleu(
      refs_, hyps_, weights=(0.25, 0.25, 0.25, 0.25))

    # pprint(bleu_scores)
    scores['b1'] = float(bleu_scores['bleu_1'])
    scores['b2'] = float(bleu_scores['bleu_2'])
    scores['b3'] = float(bleu_scores['bleu_3'])
    scores['b4'] = float(bleu_scores['bleu_4'])

    ## inter-sentence rouge 

    ## inter-sentence bleu 
    return scores

  def test_model(self, model, dataset, ckpt_e):
    """Test the model on the dev and test dataset"""
    if(self.model_name == 'latent_temp_crf'):
      self.template_manager.load(ckpt_e)

    num_batches = dataset.num_batches('train', self.batch_size)
    self.train_schedule_init(num_batches, self.start_epoch, self.num_epoch)
    n_iter = (ckpt_e + 1) * num_batches

    _, scores = self.validate(model, dataset, ckpt_e, n_iter, 'dev')
    print('dev scores')
    pprint(scores)
    _, scores = self.validate(model, dataset, ckpt_e, n_iter, 'test')
    print('test scores')
    pprint(scores)
    return 
