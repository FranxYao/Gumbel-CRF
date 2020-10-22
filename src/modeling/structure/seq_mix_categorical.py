
import torch
import copy

import numpy as np 
import torch.nn as nn
import torch.nn.functional as F

from .. import torch_model_utils as tmu

class SeqMixCat(nn.Module):
  def __init__(self, config):
    super(SeqMixCat, self).__init__()
    return 

  def rsample(self, log_prob):
    log_prob[:, 1:] += log_prob[:, :-1].clone()
    log_prob[:, :-1] += log_prob[:, 1:].clone()
    prob = F.softmax(log_prob, dim=-1)
    log_prob = torch.log(prob + 1e-10)

    relaxed_sample = tmu.reparameterize_gumbel(log_prob, tau)
    sample = torch.argmax(sample, dim=-1) # [batch, max_len]
    return sample, relaxed_sample
