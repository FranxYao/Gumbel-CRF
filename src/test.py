import argparse
import os 

import sys 
import shutil
import torch

from datetime import datetime

from modeling import torch_model_utils as tmu

from controller import Controller
from config import Config

from main import define_argument, set_argument

from modeling.latent_temp_crf import LatentTemplateCRF
from modeling.latent_temp_crf_rl import LatentTemplateCRFRL

from data_utils.dataset_e2e import DatasetE2E
from data_utils.dataset_ptb import DatasetPTB
from data_utils.dataset_mscoco import DatasetMSCOCO

config = Config()
args = define_argument(config)

args.model_version = 't.0'
args.dataset = 'e2e'
args.task = 'density'
args.gpu_id = '7'
args.z_sample_method = 'gumbel_ffbs'
args.z_beta = 1e-4
args.z_gamma = 0.
args.latent_vocab_size = 20 
args.auto_regressive = False

config = set_argument(config, args)

# dataset
if(config.dataset == 'e2e'):
  dataset = DatasetE2E(config)
  dataset.build()
  config.key_vocab_size = dataset.key_vocab_size
elif(config.dataset == 'ptb'):
  dataset = DatasetPTB(config)
  dataset.build()
elif(config.dataset == 'mscoco'):
  dataset = DatasetMSCOCO(config)
  dataset.build()
else: 
  raise NotImplementedError('dataset %s not implemented' % config.dataset)
config.vocab_size = dataset.vocab_size
# debug
with open(config.output_path + 'id2word.txt', 'w') as fd:
  for i in dataset.id2word: fd.write('%d %s\n' % (i, dataset.id2word[i]))


model = LatentTemplateCRFRL(config)
model.to(config.device)
tmu.print_params(model)
# return 

batch = dataset.next_batch('train', 10)
sentences = torch.from_numpy(batch['sent_dlex']).to(config.device)
x_lambd = 0.
tau=1.
post_process=False 
debug=False
loss, out_dict = model(
  keys=torch.from_numpy(batch['keys']).to(config.device),
  vals=torch.from_numpy(batch['vals']).to(config.device),
  sentences=sentences,
  sent_lens=torch.from_numpy(batch['sent_lens']).to(config.device),
  tau=tau, 
  x_lambd=x_lambd,
  num_sample=5, 
  post_process=post_process,
  debug=debug, 
  return_grad=False
  )