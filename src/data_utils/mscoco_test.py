"""Testing MSCOCO dataset class"""

from data_utils.dataset_mscoco import DatasetMSCOCO
from config import Config
import numpy as np 

config = Config()
config.dataset = 'mscoco'
config.model_name = 'latent_temp_crf'
config.max_dec_len = config.max_dec_len[config.dataset]
config.max_sent_len = config.max_sent_len[config.dataset]
config.max_bow_len = config.max_bow_len[config.dataset]

dset = DatasetMSCOCO(config)
dset.build()

batch_size = config.batch_size
setname = 'test'
batch = dset.next_batch(setname, batch_size)
dset.print_batch(batch, print_all=True)

num_batch = dset.num_batches(setname, 100)
for bi in range(num_batch):
  batch = dset.next_batch(setname, batch_size)
  batch_size= batch['keys'].shape[0]
  for i in range(batch_size):
    if(np.all(batch['keys'][i] == 0)): 
      print(bi, i, batch['keys'][i])
      print(bi, i, batch['mem_lens'][i])
  

# batch = dset.next_batch(setname, batch_size)
# dset.print_batch(batch, print_all=True)

## model 
from modeling.latent_temp_crf import LatentTemplateCRF
config.vocab_size = dset.vocab_size
model = LatentTemplateCRF(config)
model.to('cuda')

import torch 
from torch.optim import Adam, SGD, RMSprop
from torch.nn.utils.clip_grad import clip_grad_norm_
optimizer = Adam(model.parameters(), lr=config.learning_rate)

tau=1.0
x_lambd=0.0
post_process=True

for i in range(40):
  batch = dset.next_batch(setname, batch_size)
  model.zero_grad()
  loss, out_dict = model(
    keys=torch.from_numpy(batch['keys']).to(config.device),
    vals=torch.from_numpy(batch['vals']).to(config.device),
    sentences=torch.from_numpy(batch['sentences']).to(config.device),
    sent_lens=torch.from_numpy(batch['sent_lens']).to(config.device),
    tau=tau, 
    x_lambd=x_lambd,
    post_process=post_process
    )
  loss.backward()
  clip_grad_norm_(model.parameters(), config.max_grad_norm)
  optimizer.step()
  print('.')



def dist_key(k1, k2, pad_id=0):
  """Calculate the distance between two keys"""
  k1_ = set(list(k1))
  k1_ -= set([pad_id])
  k2_ = set(list(k2))
  k2_ -= set([pad_id])

  d = 1 - len(k1_ & k2_) / float(len(k1_ | k2_))
  return d
