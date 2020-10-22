

import csv
import numpy as np 

from nltk import word_tokenize
from tqdm import tqdm

from collections import defaultdict
from .dataset_base import DatasetBase
from . import nlp_pipeline as nlpp

def normalize_set(dset, word2id, max_sent_len, max_mem_len):
  """Normalize the train/ dev/ test set
  
  Args:
    dset: the output of the `read_data` function
    word2id: the output of the `build_vocab` function
    max_sent_len: an integer
    max_mem_len: an integer
  """
  data_sents = [d[1] for d in dset]
  data_templates = [d[2] for d in dset]
  set_keys, set_vals = [], []
  for tb, _, _ in dset:
    keys = [k for k, _ in tb]
    set_keys.append(keys)
    vals = [v for _, v in tb]
    set_vals.append(vals)
  set_keys, set_lens = nlpp.normalize(
    set_keys, word2id, max_mem_len, add_start_end=False)
  set_vals, _ = nlpp.normalize(
    set_vals, word2id, max_mem_len, add_start_end=False)
  sentences, sent_lens = nlpp.normalize(data_sents, word2id, max_sent_len)
  templates, _ = nlpp.normalize(data_templates, word2id, max_sent_len)
  return set_keys, set_vals, set_lens, sentences, templates, sent_lens

def read_data(dpath):
  """Read the raw e2e data
  
  Args:
    dpath: path to the .csv file

  Returns:
    dataset: a list of (tb, s, st) triple
      tb = the table, a list of (k, v) tuple
        k = the key 
        v = the value
      s = the sentence
      st = the sentence template 
      all characters are changed to lower
  """
  print('reading %s' % dpath)
  with open(dpath) as fd:
    reader = csv.reader(fd)
    lines = [l for l in reader]
    lines = lines[1:]

  dataset = []
  for l in tqdm(lines):
    t = l[0].lower().split(', ')
    t = [(ti.split('[')[0], ti.split('[')[1][:-1]) for ti in t]
    t_ = []
    for k, v in t:
      for i, vi in enumerate(v.split()):
        t_.append(('_' + k.replace(' ', '_') + '_' + str(i), vi))
    t = t_

    s = l[1].lower()
    # Tokenize a string to split off white spaces and punctuations other than 
    # periods
    s = word_tokenize(s)

    st = []
    for w in s:
      in_table = False
      for k, v in t:
        if(w == v):
          st.append(k)
          in_table = True
          break
      if(in_table == False):
        st.append(w)
    
    dataset.append((t, s, st))
  print("%d cases" % len(dataset))
  return dataset

def prepare_inference(keys, vals, sents, pad_id=0):
  """Prepare the data format for inference
  
  Args:
    key:
    vals: 
    sents:

  Returns: 
    keys_inf:
    vals_inf:
    references: 
  """
  keys_inf, vals_inf, mem_lens, references = [], [], [], []
  i = 0
  for k, v, s in zip(keys, vals, sents):
    if(i == 0):
      r = [s[1:]]
      prev_k = k
      prev_v = v
      i += 1
    else:
      k_ = ' '.join(str(ki) for ki in k)
      kp = ' '.join(str(ki) for ki in prev_k)
      if(k_ == kp): r.append(s[1:])
      else: 
        keys_inf.append(prev_k)
        vals_inf.append(prev_v)
        mem_lens.append(np.sum(np.array(prev_k) != pad_id))
        references.append(r)

        r = [s[1:]]
        prev_k = k
        prev_v = v

  keys_inf.append(prev_k)
  vals_inf.append(prev_v)
  mem_lens.append(np.sum(prev_k != pad_id))
  references.append(r)
  return keys_inf, vals_inf, mem_lens, references

class DatasetE2E(DatasetBase):

  def __init__(self, config):
    super(DatasetE2E, self).__init__()
    self.model_name = config.model_name
    self.task = config.task
    self.data_path = config.data_path['e2e']
    self.word2id = config.word2id
    self.id2word = config.id2word
    self.pad_id = config.pad_id
    self.key2id = {}
    self.id2key = {}
    self.max_sent_len = config.max_sent_len
    self.max_mem_len = config.max_mem_len
    self.max_bow_len = config.max_bow_len
    self.latent_vocab_size = config.latent_vocab_size

    # model configuration 
    self.auto_regressive = config.auto_regressive

    self._dataset = {"train": None, "dev": None, "test": None}
    self._ptr = {"train": 0, "dev": 0, "test": 0}
    self._reset_ptr = {"train": False, "dev": False, "test": False}
    return 

  @property
  def vocab_size(self): return len(self.word2id)

  @property
  def key_vocab_size(self): return len(self.key2id)

  def dataset_size(self, setname):
    return len(self._dataset[setname]['keys'])

  def num_batches(self, setname, batch_size):
    setsize = self.dataset_size(setname)
    num_batches = setsize // batch_size + 1
    return num_batches

  # def _update_ptr(self, setname, batch_size):
  #   ptr = self._ptr[setname]
  #   ptr += batch_size
  #   if(ptr == self.dataset_size(setname)): 
  #     ptr = 0
  #   if(ptr + batch_size > self.dataset_size(setname)):
  #     ptr = self.dataset_size(setname) - batch_size
  #   self._ptr[setname] = ptr
  #   return 

  def _update_ptr(self, setname, batch_size):
    if(self._reset_ptr[setname]):
      ptr = 0
      self._reset_ptr[setname] = False
    else: 
      ptr = self._ptr[setname]
      ptr += batch_size
      if(ptr + batch_size >= self.dataset_size(setname)):
        self._reset_ptr[setname] = True
    self._ptr[setname] = ptr
    return 

  def build(self):
    """Build the dataset"""
    max_sent_len = self.max_sent_len
    max_mem_len = self.max_mem_len

    ## read the sentences
    trainset = read_data(self.data_path['train'])
    devset = read_data(self.data_path['dev'])
    testset = read_data(self.data_path['test'])

    ## build vocabulary 
    train_sents = [t[1] for t in trainset]
    word2id, id2word, _ = nlpp.build_vocab(train_sents, 
      word2id=self.word2id, id2word=self.id2word, vocab_size_threshold=1)
    train_keys = []
    for tb, _, _ in trainset:
      keys = [k for k, _ in tb]
      train_keys.extend(keys)
    # join key vocab and the word vocab, but keep a seperate key dict
    self.word2id, self.id2word, self.key2id, self.id2key =\
      nlpp.extend_vocab_with_keys(word2id, id2word, train_keys)

    ## normalize the dataset 
    (train_keys, train_vals, train_mem_lens, train_sentences, train_templates, 
      train_sent_lens) = normalize_set(
        trainset, self.word2id, max_sent_len, max_mem_len)
    train_bow = [nlpp.sent_to_bow(s, self.max_bow_len) for s in train_sentences]
    (dev_keys, dev_vals, dev_mem_lens, dev_sentences, dev_templates, 
      dev_sent_lens) = normalize_set(
        devset, self.word2id, max_sent_len, max_mem_len)
    dev_bow = [nlpp.sent_to_bow(s, self.max_bow_len) for s in dev_sentences]
    (test_keys, test_vals, test_mem_lens, test_sentences, test_templates, 
      test_sent_lens) = normalize_set(
        testset, self.word2id, max_sent_len, max_mem_len)
    test_bow = [nlpp.sent_to_bow(s, self.max_bow_len) for s in test_sentences]

    ## Prepare the inference format
    dev_keys_inf, dev_vals_inf, dev_lens_inf, dev_references =\
      prepare_inference(dev_keys, dev_vals, dev_sentences)
    print('%d processed dev cases' % len(dev_keys_inf))
    test_keys_inf, test_vals_inf, test_lens_inf, test_references =\
      prepare_inference(test_keys, test_vals, test_sentences)
    print('%d processed test cases' % len(test_keys_inf))

    ## finalize
    self._dataset = { "train": { 
                        'sentences': train_sentences,
                        'sent_bow': train_bow, 
                        'templates': train_templates,
                        'sent_lens': train_sent_lens,
                        'keys': train_keys,
                        'vals': train_vals,
                        'mem_lens': train_mem_lens}, 
                      "dev_casewise": { 
                        'sentences': dev_sentences,
                        'sent_bow': dev_bow, 
                        'templates': dev_templates,
                        'sent_lens': dev_sent_lens,
                        'keys': dev_keys,
                        'vals': dev_vals,
                        'mem_lens': dev_mem_lens}, 
                      'dev': {
                        'keys': dev_keys_inf,
                        'vals': dev_vals_inf,
                        'mem_lens': dev_lens_inf, 
                        'references': dev_references},
                      "test_casewise": { 
                        'sentences': test_sentences,
                        'sent_bow': test_bow, 
                        'templates': test_templates,
                        'sent_lens': test_sent_lens,
                        'keys': test_keys,
                        'vals': test_vals,
                        'mem_lens': test_mem_lens},
                      'test': {
                        'keys': test_keys_inf,
                        'vals': test_vals_inf,
                        'mem_lens': test_lens_inf, 
                        'references': test_references}
                      }
    return 

  def next_batch_train(self, setname, ptr, batch_size):
    sentences = self._dataset[setname]['sentences'][ptr: ptr + batch_size]
    sent_bow = self._dataset[setname]['sent_bow'][ptr: ptr + batch_size]
    templates = self._dataset[setname]['templates'][ptr: ptr + batch_size]
    sent_lens = self._dataset[setname]['sent_lens'][ptr: ptr + batch_size]
    keys = self._dataset[setname]['keys'][ptr: ptr + batch_size]
    vals = self._dataset[setname]['vals'][ptr: ptr + batch_size]
    mem_lens = self._dataset[setname]['mem_lens'][ptr: ptr + batch_size]

    batch = {'sentences': np.array(sentences),
             'sent_bow': np.array(sent_bow),
             'sent_dlex': np.array(templates), # delexicalized sentences
             'sent_lens': np.array(sent_lens), # sent_len + _GOO 
             'keys': np.array(keys),
             'vals': np.array(vals),
             'mem_lens': np.array(mem_lens)}
    return batch

  def next_batch_infer(self, setname, ptr, batch_size):
    keys = self._dataset[setname]['keys'][ptr: ptr + batch_size]
    vals = self._dataset[setname]['vals'][ptr: ptr + batch_size]
    mem_lens = self._dataset[setname]['mem_lens'][ptr: ptr + batch_size]
    references = self._dataset[setname]['references'][ptr: ptr + batch_size]

    batch = {'keys': np.array(keys),
             'vals': np.array(vals),
             'mem_lens': np.array(mem_lens), 
             'references': references}
    return batch

  def next_batch(self, setname, batch_size):
    """Get next batch 
    
    Args:
      setname: 'train', 'dev', or 'test'
      batch_size: an integer

    Returns:
      batch: type=dict
      batch['sentences']
      batch['templates']
      batch['sent_lens']
      batch['keys']
      batch['vals']
      batch['mem_lens']
    """
    ptr = self._ptr[setname]

    if(self.model_name in ['rnnlm', 'autodecoder']):
      if(setname == 'train'):
        batch = self.next_batch_train(setname, ptr, batch_size)
      else:
        batch = self.next_batch_train(setname + '_casewise', ptr, batch_size)
    elif(self.model_name.startswith('latent_temp_crf')):
      if(setname == 'train'):
        batch = self.next_batch_train(setname, ptr, batch_size)
      else:
        if(self.task == 'density'):
          batch_c = self.next_batch_train(setname + '_casewise', ptr, batch_size)
          batch_i = self.next_batch_infer(setname, ptr, batch_size)
          batch = (batch_c, batch_i)
        elif(self.task == 'generation'):
          batch = self.next_batch_infer(setname, ptr, batch_size)
        else: 
          raise NotImplementedError(self.task)
    else: 
      if(setname == 'train'):
        batch = self.next_batch_train(setname, ptr, batch_size)
      else:
        batch = self.next_batch_infer(setname, ptr, batch_size)
    
    self._update_ptr(setname, batch_size)
    return batch

  def dump_for_test(self):
    out_path = '../data/e2e-dataset/'
    batch_size = 100
    for setname in ['dev', 'test']:
      num_batches = self.num_batches(setname, batch_size)
      with open(out_path + setname + '.txt', 'w') as fd: 
        for _ in range(num_batches):
          batch = self.next_batch(setname, batch_size)
          for r in batch['references']:
            for ri in r:
              fd.write(self.decode_sent(ri, add_eos=False) + '\n')
            fd.write('\n')

      with open(out_path + setname + '_input.txt', 'w') as fd: 
        for _ in range(num_batches):
          batch = self.next_batch(setname, batch_size)
          for k, v in zip(batch['keys'], batch['vals']):
            for ki, vi in zip(k, v):
              if(ki == self.pad_id): break
              fd.write('%s: %s\n' % (self.id2word[ki], self.id2word[vi]))
            fd.write('\n')
    return 

  def decode_sent(self, sent, sent_len=-1, prob=None, add_eos=True):
    """Decode the sentence from id to string"""
    s_out = []
    is_break = False
    for wi, wid in enumerate(sent[:sent_len]):
      if(is_break): break
      w = self.id2word[wid]
      if(w == "_EOS"): 
        is_break = True
      s_out.append(w)
      if(prob is not None): s_out.append("(%.3f) " % prob[wi])
    if(add_eos == False): s_out = s_out[:-1]
    return " ".join(s_out)

  def decode_sent_w_state(self, sent, state, sent_len=-1):
    """Decode the sentence, gather words with the same states"""
    s_out = ''
    is_break = False
    prev_state = -1
    for wi, wid in enumerate(sent[:sent_len]):
      # if(is_break): break
      w = self.id2word[wid]
      if(w == "_EOS"): break
      si = state[wi]
      if(si != prev_state):
        if(s_out != ''): s_out += ']' + str(prev_state) + ' ' + '['
        else: s_out += '['
      else: s_out += ' '
      s_out += w
      prev_state = si
    s_out += ']' + str(prev_state)
    return s_out
    
  def decode_sent_w_adapt_state(self, sent, sent_seg, state, state_len):
    """Decode the sentence, gather words with the same states

    This implementation also enables us to check if a sentence is ended by EOS 
    or the end of template by printing out EOS explicitly
    """
    s_out = '['
    is_break = False
    prev_state = -1
    k = 0 
    # print('sent_seg.shape:', sent_seg.shape)
    for wi, (wid, si) in enumerate(zip(sent, sent_seg)):
      # if(is_break): break
      w = self.id2word[wid]
      s_out += w
      if(w == "_EOS"): break
      if(si == 1):
        s_out += ']' + str(state[k]) 
        k += 1
        if(k == state_len): break
        else: s_out += ' ' + '['
      else: s_out += ' '
    
    if(k != state_len):
      s_out += ']' + str(state[k])
    return s_out

  def decode_sent_with_bow(self, sent, bow, bow_prob, sent_len=-1):
    s_out = ''
    for w, bow, bp in zip(
      sent[: sent_len], bow[: sent_len], bow_prob[: sent_len]):
      s_out += '%s: %s %.3f | %s %.3f | %s %.3f \n' % (self.id2word[w], 
        self.id2word[bow[0]], bp[0],
        self.id2word[bow[1]], bp[1],
        self.id2word[bow[2]], bp[2])
    return s_out

  def state_length_statistics(self, z_sample_ids, sent_len):
    stats = defaultdict(int)
    for i in range(z_sample_ids.shape[0]):
      l = sent_len[i]
      z = z_sample_ids[i]
      prev_state = -1
      current_chunk_len = 1
      for zi, zs in enumerate(z[: l]):
        if(zs != prev_state):
          if(zi == 0):
            pass 
          else:
            stats[current_chunk_len] += 1
            stats[0] += 1
            current_chunk_len = 1
        else: 
          current_chunk_len += 1
        prev_state = zs
      stats[current_chunk_len] += 1
      stats[0] += 1

    avg = 0
    cnt = 0
    for i in stats:
      if(i == 0): continue
      else: 
        avg += i * stats[i]
        cnt += stats[i]
    avg = avg / float(cnt)
    return stats, avg

  def print_inspect(self, inspect, batch, model_name, do_not_print=False):
    """Print the model inspection, for monitoring training"""
    out = ''

    if('z_sample_ids' in inspect): 
      z_sample_ids = inspect['z_sample_ids']
      out += 'z_sample_ids\n'
    else: z_sample_ids = None
    if('z_topk' in inspect): z_topk = inspect['z_topk']
    if('vae_predictions' in inspect): predictions = inspect['vae_predictions']
    if('train_predictions_stepwise' in inspect): 
      train_predictions_stepwise = inspect['train_predictions_stepwise']
    if('train_predictions' in inspect):
      train_predictions = inspect['train_predictions']
    if('train_post_predictions' in inspect):
      post_predictions = inspect['train_post_predictions']
    if('switch_g_prob' in inspect): switch_g_prob = inspect['switch_g_prob']
    if('dec_g' in inspect): dec_g = inspect['dec_g']
    if('bow_sample_ids' in inspect): bow_sample_ids = inspect['bow_sample_ids']
    if('bow_step_topk' in inspect): bow_step_topk = inspect['bow_step_topk']
    if('bow_step_topk_prob' in inspect): 
      bow_step_topk_prob = inspect['bow_step_topk_prob']
    if('dec_lens' in inspect): dec_lens = inspect['dec_lens']
    if('dec_targets' in inspect): dec_targets = inspect['dec_targets']

    sent_lens = batch['sent_lens']
    sentences = batch['sentences']
    sent_dlex = batch['sent_dlex']

    if(model_name in [
      'latent_temp_crf_vae', 'latent_temp_crf', 
      'latent_temp_crf_ar',
      'latent_temp_crf_ts', 'latent_temp_crf_rl'
      ] and z_sample_ids is not None
      ):
      out += 's[0] tag sampled: \n'
      out += '          ' +\
        self.decode_sent_w_state(batch['sentences'][0], z_sample_ids[0]) + '\n'
      # out += 's[0] sent:'
      # out += self.decode_sent(batch['sentences'][0]) + '\n'
      out += 's[0] sent_dlex:'
      out += self.decode_sent(batch['sent_dlex'][0]) + '\n'
      out += '\n\n'

      out += 's[1] tag sampled: \n'
      out += '          ' +\
        self.decode_sent_w_state(batch['sentences'][1], z_sample_ids[1]) + '\n'
      # out += 's[1] sent:'
      # out += self.decode_sent(batch['sentences'][1]) + '\n'
      out += 's[1] sent_dlex:'
      out += self.decode_sent(batch['sent_dlex'][1]) + '\n'
      out += '\n\n'

      latent_vocab_stats = np.zeros(self.latent_vocab_size).astype(int)
      for i in range(self.latent_vocab_size):
        latent_vocab_stats[i] = int(np.sum(z_sample_ids == i))
      print('latent state stats:')
      print(latent_vocab_stats)

    if(do_not_print): return out
    else: print(out)
    return out

  def post_process_sentence(self, keys, vals, sent):
    """Post processing single sentence"""
    sent_ = np.zeros_like(sent)
    for wi, w in enumerate(sent):
      for k, v in zip(keys, vals):
        if(w == k): 
          w = v
          break
      sent_[wi] = w
    return sent_
  
  def post_process(self, batch, out_dict):
    """Post processing the prediction, substitute keys with values"""
    if('predictions' in out_dict):
      predictions = out_dict['predictions']
      predictions_ = np.zeros_like(predictions)
      for bi in range(predictions.shape[0]):
        keys_bi = batch['keys'][bi]
        vals_bi = batch['vals'][bi]
        predictions_[bi] = self.post_process_sentence(
          keys_bi, vals_bi, predictions[bi])
      out_dict['predictions'] = predictions_

    if('post_predictions' in out_dict):
      post_predictions = out_dict['post_predictions']
      post_predictions_ = np.zeros_like(post_predictions)
      for bi in range(post_predictions.shape[0]):
        keys_bi = batch['keys'][bi]
        vals_bi = batch['vals'][bi]
        post_predictions_[bi] = self.post_process_sentence(
          keys_bi, vals_bi, post_predictions[bi])
      out_dict['post_predictions'] = post_predictions_

    if('predictions_all' in out_dict):
      predictions_all = out_dict['predictions_all']
      predictions_all_ = np.zeros_like(predictions_all)
      for bi in range(predictions_all.shape[0]):
        keys_bi = batch['keys'][bi]
        vals_bi = batch['vals'][bi]
        for si in range(predictions_all.shape[1]):
          predictions_all_[bi][si] = self.post_process_sentence(
            keys_bi, vals_bi, predictions_all[bi][si])
      out_dict['predictions_all'] = predictions_all_

    if('post_predictions_all' in out_dict):
      predictions_all = out_dict['post_predictions_all']
      predictions_all_ = np.zeros_like(predictions_all)
      for bi in range(predictions_all.shape[0]):
        keys_bi = batch['keys'][bi]
        vals_bi = batch['vals'][bi]
        for si in range(predictions_all.shape[1]):
          predictions_all_[bi][si] = self.post_process_sentence(
            keys_bi, vals_bi, predictions_all[bi][si])
      out_dict['post_predictions_all'] = predictions_all_
    return 

  def print_batch(self, batch, 
    out_dict=None, model_name=None, fd=None, fd_full=None):
    """Print out a test batch"""
    if(fd is None): 
      print_range = np.random.choice(
        len(batch['keys']), 5, replace=False)
    else: print_range = range(len(batch['keys']))

    out = '' 
    pred_out = ''
    post_out = ''

    print('debug: print_range %d' % len(print_range))
    for i in print_range:
      out += 'mem:\n'
      for k, v in zip(batch['keys'][i][: batch['mem_lens'][i]], 
        batch['vals'][i][: batch['mem_lens'][i]]):
        out += self.id2word[k] + ': ' + self.id2word[v] + ' | '
      out += '\n'

      if('sentences' in batch):
        out += 'sentence:\n'
        out += self.decode_sent(batch['sentences'][i]) + '\n'

      if('references' in batch):
        out += 'references\n'
        for j in range(len(batch['references'][i])):
          out += '%d: ' % j
          out += self.decode_sent(batch['references'][i][j]) + '\n'

      if(out_dict is not None):
        out += 'predictions:\n'
        if(model_name.startswith('latent_temp')):
          s_out = self.decode_sent_w_state(
            out_dict['predictions'][i], 
            out_dict['predictions_z'][i]) + '\n'
          out += s_out

          s_out = self.decode_sent(out_dict['predictions'][i]) + '\n'
          pred_out += s_out
          
      out += '\n\n'

    out += '\n'
    if(fd is not None): fd.write(pred_out)
    else: print(out)

    if(fd_full is not None): fd_full.write(out + '\n\n')
    return 
  