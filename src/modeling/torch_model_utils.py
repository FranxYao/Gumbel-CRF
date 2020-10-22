"""Torch model/ missing utils

the missing utility library for pytorch

Tensor operations:
* `to_np`
* `length_to_mask`
* `ind_to_one_hot`
* `bow_to_one_hot`
* `seq_to_lens`
* `find_index`
* `seq_ends`
* `lens_to_mask`
* `reverse_sequence`
* `gather_last`
* `batch_index_select`
* `batch_index_put`
* `batch_repeat`

Probability:
* `sample_gumbel`
* `reparameterize_gumbel`
* `seq_gumbel_encode` # needs update
* `reparameterize_gaussian`
* `entropy`
* `kl_divergence`
* `js_divergence`

Model operations:
* `load_partial_state_dict`
* `print_params`
* `print_grad`
"""

import numpy as np 

import torch
import torch.nn.functional as F

from torch.nn.parameter import Parameter
from collections import OrderedDict

def to_np(x):
  """Export a tensor to numpy"""
  return x.detach().cpu().numpy()

def length_to_mask(length, max_len):
  """
  True = 1 = not masked, False = 0 = masked

  Args:
    length: type=torch.tensor(int), size=[batch]
    max_len: type=int

  Returns:
    mask: type=torch.tensor(bool), size=[batch, max_len]
  """
  batch_size = length.shape[0]
  device = length.device
  mask = torch.arange(max_len, dtype=length.dtype)\
    .expand(batch_size, max_len).to(device) < length.unsqueeze(1)
  return mask

def mask_by_length(A, lens, mask_id=0.):
  """mask a batch of seq tensor by length
  
  Args:
    A: type=torch.tensor(), size=[batch, max_len, *]
    lens: type=torch.tensor(ing), size=[batch]
    mask_id: type=float

  Returns
    A_masked: type=torch.tensor(float), note the for whatever input, the output 
      type would be casted to float
  """
  mask = length_to_mask(lens, A.size(1))
  target_size = list(mask.size()) + [1] * (len(A.size()) - 2)
  mask = mask.view(target_size)

  A_masked = A.float() * mask.float() + A.float() * (1 - mask.float()) * mask_id
  return A_masked

def ind_to_one_hot(ind, max_len):
  """Index to one hot representation

  Args:
    length: type=torch.tensor(int), size=[batch]
    max_len: type=int

  Returns:
    one_hot: type=torch.tensor(bool), size=[batch, max_len]

  Note: 
    by default, `one_hot.dtype = ind.dtype`, and there is no constraint on 
    `ind.dtype`. So it is also possible to pass `ind` with float type 
  """
  device = ind.device
  batch_size = ind.shape[0]
  one_hot = torch.arange(max_len, dtype=ind.dtype)\
    .expand(batch_size, max_len).to(device) == (ind).unsqueeze(1)
  return one_hot

def bow_to_one_hot(bow, vocab_size, pad_id=0):
  """Bag of words to one hot representation

  Args:
    bow: type=torch.tensor(int), size=[batch, max_bow]
    vocab_size: type=int
    pad_id: type=int

  Returns:
    one_hot: type=torch.tensor(int), size=[batch, vocab_size]
  """
  device = bow.device
  batch_size = bow.shape[0]
  bow = bow.view(-1).unsqueeze(1) # [batch * bow, 1]
  one_hot = (bow == torch.arange(vocab_size).to(device)\
    .reshape(1, vocab_size)).float()
  one_hot = one_hot.view(batch_size, -1, vocab_size)
  one_hot.index_fill_(
    dim=2, index=torch.tensor([pad_id]).to(device), value=0)
  one_hot = one_hot.sum(dim=1)
  return one_hot

def seq_to_lens(seq, pad_id=0):
  """Calculate sequence length
  
  Args:
    seq: type=torch.tensor(long), shape=[*, max_len]
    pad_id: pad index. 

  Returns:
    lens: type=torch.tensor(long), shape=[*]
  """
  lens = (seq != pad_id).sum(dim=-1).type(torch.long)
  return lens

def find_index(seq, val):
  """Find the first location index of a value 
  if there is no such value, return -1
  
  Args:
    seq: type=torch.tensor(long), shape=[batch, max_len]
    val: type=int 

  Returns:
    lens: type=torch.tensor(long), shape=[batch]
  """
  device = seq.device
  s_ = (seq == val).float()
  seq_len = seq.size(-1)
  ind_ = torch.arange(seq_len).view(1, seq_len) + 1
  ind_ = ind_.to(device).float()
  s = (1 - s_) * 1e10 + s_ * ind_
  _, index = torch.min(s, dim=-1)
  index = index.long()
  not_find = (s_.sum(-1) == 0)
  index.masked_fill_(not_find, -1)
  return index

def seq_ends(seq, end_id):
  """Calculate where the sequence ends
  if there is not end_id, return the last index 
  
  Args:
    seq: type=torch.tensor(long), shape=[batch, max_len]
    end_id: end index. 

  Returns:
    ends_at: type=torch.tensor(long), shape=[batch]
  """
  ends_at = find_index(seq, end_id)
  max_len = seq.size(1) - 1
  ends_at[ends_at == -1] = max_len
  return ends_at

def reverse_sequence(seq, seq_lens):
  """Reverse the sequence

  Examples:

  seq = [[1, 2, 3, 4, 5], [6, 7 ,8, 9, 0]], seq_lens = [3, 4]
  reverse_sequence(seq, seq_lens) = [[3, 2, 1, 4, 5], [9, 8, 7, 6, 0]]

  seq = [[[1, 1], [2, 2], [3, 3], [4, 4], [5, 5]], 
         [[6, 6], [7, 7], [8, 8], [9, 9], [0, 0]]], 
  seq_lens = [3, 4]
  reverse_sequence(seq, seq_lens) = 
    [[[3, 3], [2, 2], [1, 1], [4, 4], [5, 5]], 
     [[9, 9], [8, 8], [7, 7], [6, 6], [0, 0]]]
  
  Args: 
    seq: size=[batch, max_len, *]
    seq_lens: size=[batch]

  Returns:
    reversed_seq
  """
  batch = seq.size(0)
  reversed_seq = seq.clone()
  for i in range(batch):
    ind = list(range(seq_lens[i]))
    ind.reverse()
    reversed_seq[i,:seq_lens[i]] = seq[i, ind]
  return reversed_seq

def gather_last(seq, seq_lens):
  """Gather the last element of a given sequence"""
  return batch_index_select(seq, seq_lens - 1)

def batch_index_select(A, ind):
  """Batched index select
  
  Args:
    A: size=[batch, num_class, *] 
    ind: size=[batch, num_select] or [batch]

  Returns:
    A_selected: size=[batch, num_select, *] or [batch, *]
  """  
  batch_size = A.size(0)
  num_class = A.size(1)
  A_size = list(A.size())
  device = A.device
  A_ = A.clone().reshape(batch_size * num_class, -1)
  if(len(ind.size()) == 1): 
    batch_ind = (torch.arange(batch_size) * num_class)\
      .type(torch.long).to(device)
    ind_ = ind + batch_ind
    A_selected = torch.index_select(A_, 0, ind_)\
      .view([batch_size] + A_size[2:])
  else:
    batch_ind = (torch.arange(batch_size) * num_class)\
      .type(torch.long).to(device)
    num_select = ind.size(1)
    batch_ind = batch_ind.view(batch_size, 1)
    ind_ = (ind + batch_ind).view(batch_size * num_select)
    A_selected = torch.index_select(A_, 0, ind_)\
      .view([batch_size, num_select] + A_size[2:])
  return A_selected

def batch_index_put(A, ind, N):
  """distribute a batch of values to given locations

  Example:
    A = tensor([[0.1000, 0.9000],
                [0.2000, 0.8000]])
    ind = tensor([[1, 2],
                  [0, 3]])
    N = 5
  then:
    A_put = tensor([[0.0000, 0.1000, 0.9000, 0.0000, 0.0000],
                    [0.2000, 0.0000, 0.0000, 0.8000, 0.0000]])

  Args:
    A: size=[batch, M, *], * can be any list of dimensions
    ind: size=[batch, M]
    N: type=int

  Returns:
    A_put: size=[batch, N, *]
  """
  batch_size = A.size(0)
  M = A.size(1)
  As = list(A.size()[2:])
  device = A.device
  A_put = torch.zeros([batch_size * N] + As).to(device)
  ind_ = torch.arange(batch_size).view(batch_size, 1) * N
  ind_ = ind_.expand(batch_size, M).flatten().to(device)
  ind_ += ind.flatten()
  A_put[ind_] += A.view([batch_size * M] + As)
  A_put = A_put.view([batch_size, N] + As)
  return A_put

def batch_repeat(A, n):
  """
  Args:
    A: size=[batch, *], * can be any list of dimensions
    n: type=int

  Returns:
    A: size=[batch * n, *]
  """
  batch_size = A.size(0)
  As = list(A.size()[1:])
  A_ = A.view([batch_size, 1] + As)
  A_ = A_.repeat([1, n] + [1] * len(As))
  A_ = A_.view([batch_size * n] + As)
  return A_

def sample_gumbel(shape, eps=1e-20):
  """Sample from a standard gumbel distribution"""
  U = torch.rand(shape)
  return -torch.log(-torch.log(U + eps) + eps)

def reparameterize_gumbel(logits, tau):
  """Reparameterize gumbel sampling

  Note: gumbel reparameterization will give you sample no matter tau. tau just 
  controls how close the sample is to one-hot 
  
  Args: 
    logits: shape=[*, vocab_size]
    tau: the temperature, typically start from 1.0 and anneal to 0.01

  Returns:
    y: shape=[*, vocab_size]
  """
  y = logits + sample_gumbel(logits.size()).to(logits.device)
  return F.softmax(y / tau, dim=-1)

def seq_gumbel_encode(sample, sample_ids, embeddings, gumbel_st):
  """Encoding of gumbel sample. Given a sequence of relaxed one-hot 
  representations, return a sequence of corresponding embeddings

  TODO: drop `probs`, only use `sample`

  Args:
    sample: type=torch.tensor(torch.float), shape=[batch, max_len, vocab_size]
    sample_ids: type=torch.tensor(torch.long), shape=[batch, max_len]
    embeddings: type=torch.nn.Embeddings
    gumbel_st: type=bool, if use gumbel straight through estimator
  """
  batch_size = sample.size(0)
  max_len = sample.size(1)
  vocab_size = sample.size(2)
  if(gumbel_st):
    # straight-through version, to avoid training-inference gap 
    sample_emb = embeddings(sample_ids)
    sample_one_hot = ind_to_one_hot(
      sample_ids.view(-1), vocab_size)
    sample_one_hot =\
      sample_one_hot.view(batch_size, max_len, vocab_size)
    sample_soft = sample.masked_select(sample_one_hot)
    sample_soft = sample_soft.view(batch_size, max_len, 1)
    sample_emb *= (1 - sample_soft).detach() + sample_soft
  else:
    # original version, requires annealing in the end of training
    sample_emb = torch.matmul(
      sample.view(-1, vocab_size), embeddings.weight)
    embedding_size = sample_emb.size(-1)
    # [batch * max_len, embedding_size] -> [batch, max_len, embedding_size]
    sample_emb = sample_emb.view(
      batch_size, max_len, embedding_size)
  return sample_emb

def reparameterize_gaussian(mu, logvar):
  """Reparameterize the gaussian sample"""
  std = torch.exp(0.5 * logvar)
  eps = torch.randn_like(std)
  return mu + eps * std

def entropy(p, eps=1e-10, keepdim=False):
  """Calculate the entropy of a discrete distribution
  
  Args: 
    p: shape = [*, support_size]
  """
  ent = (-p * torch.log(p + eps)).sum(dim=-1)
  if(keepdim): return ent
  else: return ent.mean()


def kl_divergence(p0, p1, eps=1e-10):
  """Calculate the kl divergence between two distributions

  Args: 
    p0: size=[*, support_size]
    p1: size=[*, support_size]
  """
  kld = p0 * torch.log(p0 / (p1 + eps) + eps)
  kld = kld.sum(dim=-1)
  return kld

def js_divergence(p0, p1):
  """Calculate the Jensen-Shannon divergence between two distributions
  
  Args: 
    p0: size=[*, support_size]
    p1: size=[*, support_size]
  """
  p_ = (p0 + p1) / 2
  jsd = (kl_divergence(p0, p_) + kl_divergence(p1, p_)) / 2
  return jsd

def load_partial_state_dict(model, state_dict):
  """Load part of the model

  NOTE: NEED TESTING!!!

  Args:
    model: the model 
    state_dict: partial state dict
  """
  print('Loading partial state dict ... ')
  own_state = model.state_dict()
  own_params = set(own_state.keys())
  for name, param in state_dict.items():
    if name not in own_state:
      print('%s passed' % name)
      continue
    if isinstance(param, Parameter):
      # backwards compatibility for serialized parameters
      param = param.data
    print('loading: %s ' % name)
    own_params -= set(name)
    own_state[name].copy_(param)
  print('%d parameters not initialized: ' % len(own_params))
  for n in own_params: print(n)

def print_params(model):
  """Print the model parameters"""
  for name, param in model.named_parameters(): 
     print('  ', name, param.data.shape, 'requires_grad', param.requires_grad)
  return 

def print_grad(model, level='first'):
  """Print the gradient norm and std, for inspect training

  Note: the variance of gradient printed here is not the variance of a gradient 
  estimator
  """
  if(level == 'first'): print_grad_first_level(model)
  elif(level == 'second'): print_grad_second_level(model)
  else: 
    raise NotImplementedError(
      'higher level gradient inpection not implemented!')

def print_grad_first_level(model):
  """Print the gradient norm of model parameters, up to the first level name 
  hierarchy 
  """
  print('gradient of the model parameters:')

  grad_norms = OrderedDict()
  grad_std = OrderedDict()
  for name, param in model.named_parameters():
    splitted_name = name.split('.')
    first_level_name = splitted_name[0]

    if(first_level_name not in grad_norms): 
      grad_norms[first_level_name] = []
      grad_std[first_level_name] = []

    if(param.requires_grad and param.grad is not None):
      grad_norms[first_level_name].append(
        to_np(param.grad.norm()))
      grad_std[first_level_name].append(
        to_np(param.grad.var(unbiased=False)))

  for fn in grad_norms:
    if(isinstance(grad_norms[fn], list)):
      print(fn, np.average(grad_norms[fn]), np.average(grad_std[fn]))
      # print(fn, np.average(grad_norms[fn]), grad_std[fn])

  print('')
  return 

def print_grad_second_level(model):
  """Print the gradient norm of model parameters, up to the second level name 
  hierarchy 
  """
  print('gradient of the model parameters:')

  grad_norms = OrderedDict()
  grad_std = OrderedDict()
  for name, param in model.named_parameters():
    splitted_name = name.split('.')
    first_level_name = splitted_name[0]

    if(first_level_name not in grad_norms): 
      if(len(splitted_name) == 1):
        grad_norms[first_level_name] = []
        grad_std[first_level_name] = []
      else:
        grad_norms[first_level_name] = {}
        grad_std[first_level_name] = {}

    if(len(splitted_name) > 1):
      second_level_name = splitted_name[1]
      if(second_level_name not in grad_norms[first_level_name]):
        grad_norms[first_level_name][second_level_name] = []
        grad_std[first_level_name][second_level_name] = []

    if(param.requires_grad and param.grad is not None):
      # print(name, param.grad.norm(), param.grad.std())
      if(len(splitted_name) == 1):
        grad_norms[first_level_name].append(
          param.grad.norm().detach().cpu().numpy())
        grad_std[first_level_name].append(
          param.grad.std().detach().cpu().numpy())
      else: 
        grad_norms[first_level_name][second_level_name].append(
          param.grad.norm().detach().cpu().numpy())  
        grad_std[first_level_name][second_level_name].append(
          param.grad.std().detach().cpu().numpy())  

  print(grad_norms.keys())
  for fn in grad_norms:
    if(isinstance(grad_norms[fn], list)):
      print(fn, np.average(grad_norms[fn]),
        np.average(grad_std[fn]))
    else: 
      for sn in grad_norms[fn]:
        print(fn, sn, 
          np.average(grad_norms[fn][sn]),
          np.average(grad_std[fn][sn]))

  print('')
  return 
