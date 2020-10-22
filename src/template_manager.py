import numpy as np
import torch
import editdistance

from tqdm import tqdm 
from collections import defaultdict, Counter
from modeling import torch_model_utils as tmu

def _next_ngram(temp, sent, idx, n, slen):
  """Get the next ngram from a template-sentence pair"""
  if(idx == slen): return None, None, None
  ti = temp[idx]
  t = [ti]
  n_ = 1
  for i in range(idx + 1, slen):
    if(temp[i] != ti): 
      n_ += 1
      if(n_ == 2): idx_ = i
      ti = temp[i]
      t.append(ti)
    if(n_ == n + 1): break 
  if(n_ == n + 1):
    t = t[: -1]
    t = [str(ti) for ti in t]
    s = sent[idx: i]
    s = [str(si) for si in s]
    return idx_, '-'.join(t), '-'.join(s)
  elif(n_ == n):
    if(n == 1): idx_ = slen
    t = [str(ti) for ti in t]
    s = sent[idx: slen]
    s = [str(si) for si in s]
    return idx_, '-'.join(t), '-'.join(s)
  else:
    return None, None, None

def _analyse_temp(temp, sent, slen):
  """Analysis a template-sentence pair"""
  out_dict = {1: {}, 2: {}, 3: {}, 4: {}}
  for n in range(1, 5):
    idx = 0
    idx, t, w = _next_ngram(temp, sent, idx, n, slen)
    while(t != None):
      if(t not in out_dict[n]): out_dict[n][t] = [w]
      else: out_dict[n][t].append(w)
      idx, t, w = _next_ngram(temp, sent, idx, n, slen)
  return out_dict

def _get_temp_ngram_dict(templates, sentences, sent_lens):
  """Get template ngram to sentence segment statistics"""
  temp_ngram = {1: {}, 2: {}, 3: {}, 4:{}}
  temp_statistics = {1: [], 2: [], 3: [], 4:[]}
  print('Analyzing template ngram - sentence segment pairs')
  for i in tqdm(range(len(templates))):
    t, s, sl = templates[i], sentences[i], sent_lens[i]
    temp_n = _analyse_temp(t, s, sl)
    for n in temp_n: 
      temp_statistics[n].extend(list(temp_n[n].keys()))
      for k in temp_n[n]:
        if(k in temp_ngram[n]): temp_ngram[n][k].extend(temp_n[n][k])
        else: temp_ngram[n][k] = temp_n[n][k]

  for n in temp_ngram:
    temp_statistics[n] = Counter(temp_statistics[n])
    for k in temp_ngram[n]: temp_ngram[n][k] = Counter(temp_ngram[n][k])
  return temp_ngram, temp_statistics

def _temp_ngram_dict(templates, sentences, pad_id=0):
  """Build the template-ngram dictionary 

  Args:
    templates: type=np.array()
    sentences: type=np.array()

  Returns:
    temp_ngram: type=dict()
  """
  temp_ngram = {}
  for t, s in zip(templates, sentences):
    i = 0
    prev_temp = -1
    ngram = []
    for ti, wi in zip(t, s):
      if(wi == pad_id): break
      # a new template state
      if(ti != prev_temp and i != 0):
        # save previous template 
        if(prev_temp not in temp_ngram): 
          temp_ngram[prev_temp] = ['_'.join(str(ng) for ng in ngram)]
        else: temp_ngram[prev_temp].append('_'.join(str(ng) for ng in ngram))
        # update new ngram 
        ngram = [wi]
      else: 
        ngram.append(wi)
      i += 1
      prev_temp = ti
    if(prev_temp not in temp_ngram): 
      temp_ngram[prev_temp] = ['-'.join(str(ng) for ng in ngram)]
    else: temp_ngram[prev_temp].append('-'.join(str(ng) for ng in ngram))
  for k in temp_ngram:
    temp_ngram[k] = Counter(temp_ngram[k])
  return temp_ngram

# def dist_key(k1, k2, pad_id=0):
#   """Calculate the distance between two keys"""
#   k1_ = []
#   for k in k1:
#     if(k == pad_id): break
#     k1_.append(str(k))

#   k2_ = []
#   for k in k2:
#     if(k == pad_id): break 
#     k2_.append(str(k))

#   k1_ = '_'.join(k1_)
#   k2_ = '_'.join(k2_)
#   d = editdistance.eval(k1_, k2_)
#   return d

def dist_key(k1, k2, pad_id=0):
  """Calculate the distance between two keys"""
  k1_ = set(list(k1))
  k1_ -= set([pad_id])
  k2_ = set(list(k2))
  k2_ -= set([pad_id])

  d = 1 - len(k1_ & k2_) / float(len(k1_ | k2_))
  return d

def include_key(k1, k2, pad_id=0):
  """If k1 includes k2"""
  k1_, k2_ = [], []
  for k in k1:
    if(k == pad_id): break 
    k1_.append(k)
  k1 = set(k1_)
  
  for k in k2:
    if(k == pad_id): break
    k2_.append(k)
  k2 = set(k2_)
  return len(k2 - k1) == 0

class TemplateManager(object):
  def __init__(self, config, id2word):
    super(TemplateManager, self).__init__()
    self.dec_adaptive = config.dec_adaptive
    self.id2word = id2word
    self.max_dec_len = config.max_dec_len + 1
    self.output_path = config.output_path
    self.pad_id = config.pad_id
    self.device = config.device

    self.keys = None
    self.vals = None
    self.sentences = None
    self.templates = None
    self.temp_lens = None
    self.temp_ngram = None
    return 
  
  def add_batch(self, batch, out_dict):
    """add a batch of templates"""
    # TODO: index templates with keys in batch 
    batch_size = out_dict['z_sample_ids'].shape[0]
    temp_len = out_dict['z_sample_ids'].shape[1]
    templates = np.zeros([batch_size, self.max_dec_len], dtype=int)
    templates[:, :temp_len] = out_dict['z_sample_ids']
    sentences = np.zeros([batch_size, self.max_dec_len], dtype=int)
    sentences[:, :temp_len] = batch['sentences'][:, :temp_len]
    keys = batch['keys']
    vals = batch['vals']
    tlens = batch['sent_lens']
    if(self.templates is None):
      self.templates = templates
      self.sentences = sentences
      self.temp_lens = tlens
      self.keys = keys
      self.vals = vals
    else: 
      self.templates = np.concatenate([self.templates, templates], axis=0)
      self.sentences = np.concatenate([self.sentences, sentences], axis=0)
      self.temp_lens = np.concatenate([self.temp_lens, tlens], axis=0)
      self.keys = np.concatenate([self.keys, keys], axis=0)
      self.vals = np.concatenate([self.vals, vals], axis=0)
    return 

  def sample(self, num_sample):
    """Sample templates

    Args:
      num_sample: number of sample, type=int

    Returns:
      templates: type=torch.tensor(int), size=[num_sample, max_len]
      temp_lens: type=torch.tensor(int), size=[num_sample]
    """
    num_temp = self.templates.shape[0]
    sample_ind = np.random.choice(num_temp, num_sample, replace=True)
    templates = self.templates[sample_ind]
    temp_lens = self.temp_lens[sample_ind]
    sampled_keys = self.keys[sample_ind]
    sampled_vals = self.vals[sample_ind]
    sampled_sents = self.sentences[sample_ind]
    if(self.dec_adaptive): 
      templates, temp_lens = self.squeeze_templates(templates, temp_lens)
    return templates, temp_lens, sampled_keys, sampled_vals, sampled_sents

  def squeeze_templates(self, templates, temp_lens):
    """Sequeeze templates with repeated states"""
    templates_  = np.zeros_like(templates)
    temp_lens_ = []
    for i in range(templates.shape[0]):
      t = templates[i]
      l = temp_lens[i]
      k = 0
      prev_j = -1
      for j in range(l):
        if(t[j] != prev_j): 
          templates_[i][k] = t[j]
          k += 1
        prev_j = t[j]
      temp_lens_.append(k)
    temp_lens_ = np.array(temp_lens_)
    return templates_, temp_lens_

  def topk_close_to(self, keys, k, batch_close_k_ind=None):
    """Get the topk template close to the given keys

    TODO: register top k close template index in the begining 
    
    Args:
      keys: the keys as query 
      k: top k close to 

    Returns:
      close_temp:
      close_lens:
      close_keys:
      close_vals:
      close_sents:
    """
    close_temp, close_lens = [], []
    close_keys, close_vals, close_sents = [], [], []
    training_size = self.keys.shape[0]
    if(training_size > 10000):
      sampled_range = np.random.choice(training_size, 10000, replace=False)
    else: 
      sampled_range = np.array(list(range(training_size)))

    for bi, ki in enumerate(keys):
      if(batch_close_k_ind is None):
        dist_ = []
        for i in sampled_range: 
          d = dist_key(self.keys[i], ki)
          dist_.append(d)
        close_dist = np.argsort(dist_)
        close_k_ind = close_dist[: k]
        close_k_ind = sampled_range[close_k_ind]
      else:
        close_k_ind = batch_close_k_ind[bi]
      close_temp.append(self.templates[close_k_ind])
      close_lens.append(self.temp_lens[close_k_ind])
      close_keys.append(self.keys[close_k_ind])
      close_vals.append(self.vals[close_k_ind])
      close_sents.append(self.sentences[close_k_ind])
      
    close_temp = np.array(close_temp)
    close_lens = np.array(close_lens)
    close_keys = np.array(close_keys)
    close_vals = np.array(close_vals)
    close_sents = np.array(close_sents)
    if(self.dec_adaptive): 
      batch_size = close_temp.shape[0]
      close_temp = np.reshape(close_temp, [batch_size * k, -1])
      close_lens = np.reshape(close_lens, [batch_size * k])
      close_temp, close_lens = self.squeeze_templates(close_temp, close_lens)
      close_temp = np.reshape(close_temp, [batch_size, k, -1])
      close_lens = np.reshape(close_lens, [batch_size, k])
    return close_temp, close_lens, close_keys, close_vals, close_sents

  def max_close_inclusive(self, keys):
    """Get the template which keys are most close, and included by the given 
    keys.

    e.g: query = [1, 2, 3], candidate 0 = [1, 2], candidate 1 = [1, 2, 4]
    all edit distance = 1, but candidate 0 is included by query, so return 
    candidate 0 
    
    """
    close_temp = []
    temp_lens = []
    close_keys = []
    close_vals = []
    close_sents = []
    for k in keys:
      dist = 1e10
      dist_in = 1e10
      close_in_k = None

      for i in range(self.keys.shape[0]):
        train_k = self.keys[i]
        train_v = self.vals[i]
        train_s = self.sentences[i]
        train_t = self.templates[i]
        tl = self.temp_lens[i]
        dist_ = dist_key(k, train_k)

        if(include_key(k, train_k)):
          if(dist_ < dist_in):
            dist_in = dist_
            temp_in = train_t
            temp_in_l = tl
            close_in_k = train_k
            close_in_v = train_v
            close_in_s = train_s
        if(dist_ < dist):
          temp = train_t
          dist = dist_
          temp_l = tl
          close_k = train_k
          close_v = train_v
          close_s = train_s

      if(close_in_k is not None):
        close_temp.append(temp_in)
        temp_lens.append(temp_in_l)
        close_keys.append(close_in_k)
        close_vals.append(close_in_v)
        close_sents.append(close_in_s)
      else: 
        close_temp.append(temp)
        temp_lens.append(temp_l)
        close_keys.append(close_k)
        close_vals.append(close_v)
        close_sents.append(close_s)

    close_temp = np.array(close_temp)
    temp_lens = np.array(temp_lens)
    close_keys = np.array(close_keys)
    close_sents = np.array(close_sents)

    if(self.dec_adaptive): 
      close_temp, temp_lens = self.squeeze_templates(close_temp, temp_lens)
    return close_temp, temp_lens, close_keys, close_vals, close_sents

  def max_close_to(self, keys):
    """Get the template most close to the given keys"""
    close_temp = []
    temp_lens = []
    close_keys = []
    close_vals = []
    close_sents = []
    training_size = self.keys.shape[0]
    if(training_size > 10000):
      sampled_range = np.random.choice(training_size, 10000, replace=False)
    else: 
      sampled_range = list(range(training_size))
    for k in keys:
      dist = 1e10
      temp = None
      temp_l = None
      close_k = None
      for i in sampled_range:
        train_k = self.keys[i]
        train_v = self.vals[i]
        train_s = self.sentences[i]
        train_t = self.templates[i]
        tl = self.temp_lens[i]
        dist_ = dist_key(k, train_k)
        if(dist_ < dist):
          temp = train_t
          dist = dist_
          temp_l = tl
          close_k = train_k
          close_v = train_v
          close_s = train_s
      close_temp.append(temp)
      temp_lens.append(temp_l)
      close_keys.append(close_k)
      close_vals.append(close_v)
      close_sents.append(close_s)

    close_temp = np.array(close_temp)
    temp_lens = np.array(temp_lens)
    close_keys = np.array(close_keys)
    close_sents = np.array(close_sents)

    if(self.dec_adaptive): 
      close_temp, temp_lens = self.squeeze_templates(close_temp, temp_lens)
    return close_temp, temp_lens, close_keys, close_vals, close_sents

  def clear(self):
    """Clear all templates"""
    self.templates = None
    self.sentences = None
    self.temp_lens = None
    self.temp_ngram = None
    self.keys = None
    self.vals = None
    return 

  def save(self, ei):
    """Save templates to output

    Args:
      ei: epoch id
    """
    output_path = self.output_path + 'temp_e' + str(ei) + '_'
    np.save(output_path + 'templates', self.templates)
    np.save(output_path + 'sentences', self.sentences)
    np.save(output_path + 'temp_lens', self.temp_lens)
    np.save(output_path + 'keys', self.keys)
    np.save(output_path + 'vals', self.vals)
    return 

  def load(self, ei):
    load_path = self.output_path + 'temp_e' + str(ei) + '_'
    self.templates = np.load(load_path + 'templates.npy')
    self.sentences = np.load(load_path + 'sentences.npy')
    self.temp_lens = np.load(load_path + 'temp_lens.npy')
    self.keys = np.load(load_path + 'keys.npy')
    self.vals = np.load(load_path + 'vals.npy')
    return 

  def report_statistics(self, id2word, ei):
    """Print template statistics"""
    output_path = self.output_path + 'temp_e' + str(ei) + '_'
    print('Reporting template statistics, %d templates in total' 
      % self.templates.shape[0])

    # template ngram statistics 
    temp_ngram, temp_statistics = _get_temp_ngram_dict(
      self.templates, self.sentences, self.temp_lens)
    for n in temp_ngram:
      with open(output_path + '%dgram.txt' % n, 'w') as fd:
        for t, tk in temp_statistics[n].most_common():
          # t_ = [id2word[int(ti)] for ti in t.split('_')]
          fd.write(t + ' | %d\n' % tk)
          for s, sk in temp_ngram[n][t].most_common(25):
            s_ = ' '.join([id2word[int(si)] for si in s.split('-')])
            fd.write('  ' + s_ + ' | %d\n' % sk)
    return 
