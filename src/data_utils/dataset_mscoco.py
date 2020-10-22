
import nltk 
import json

import numpy as np 
import pickle
import os
import sys
sys.path.append("..")

from . import nlp_pipeline as nlpp
from template_manager import dist_key
# from ..template_manager import dist_key
from collections import Counter, defaultdict
from tqdm import tqdm
from nltk.translate.bleu_score import corpus_bleu
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from pprint import pprint 

def quora_read(file_path, bleu_baseline=False):
  """Read the quora dataset"""
  print("Reading quora raw data .. ")
  print("  data path: %s" % file_path)
  with open(file_path) as fd:
    lines = fd.readlines()
  sentence_sets = []
  for l in tqdm(lines):
    p0, p1 = l[:-1].lower().split("\t")
    sentence_sets.append([word_tokenize(p0), word_tokenize(p1)])

  if(bleu_baseline):
    print("calculating bleu ... ")
    hypothesis = [s[0] for s in sentence_sets]
    references = [s[1:] for s in sentence_sets]
    bleu = corpus_bleu(references, hypothesis)
    print("bleu on the training set: %.4f" % bleu)
  return sentence_sets

def mscoco_read_json(file_path, bleu_baseline=False):
  """Read the mscoco dataset
  Args:
    file_path: path to the raw data, a string
  Returns:
    sentence_sets: the sentence sets, a list of paraphrase lists
  """
  print("Reading mscoco raw data .. ")
  print("  data path: %s" % file_path)
  with open(file_path, "r") as fd:
    data = json.load(fd)

  print("%d sentences in total" % len(data["annotations"]))
  
  # aggregate all sentences of the same images
  image_idx = set([d["image_id"] for d in data["annotations"]])
  paraphrases = {}
  for im in image_idx: paraphrases[im] = []
  for d in tqdm(data["annotations"]):
    im = d["image_id"]
    sent = d["caption"]
    paraphrases[im].append(word_tokenize(sent))

  # filter out sentence sets size != 5 
  sentence_sets = [paraphrases[im] for im in paraphrases 
    if(len(paraphrases[im]) == 5)]

  # blue on the training set, a baseline/ upperbound
  if(bleu_baseline):
    print("calculating bleu ... ")
    hypothesis = [s[0] for s in sentence_sets]
    references = [s[1:] for s in sentence_sets]
    bleu = dict()
    bleu['1'] = corpus_bleu(
      references, hypothesis, weights=(1., 0, 0, 0))
    bleu['2'] = corpus_bleu(
      references, hypothesis, weights=(0.5, 0.5, 0, 0))
    bleu['3'] = corpus_bleu(
      references, hypothesis, weights=(0.333, 0.333, 0.333, 0))
    bleu['4'] = corpus_bleu(
      references, hypothesis, weights=(0.25, 0.25, 0.25, 0.25))
    print('bleu on the training set:')
    pprint(bleu)
  return sentence_sets

def train_dev_split(dataset_name, train_sets, train_index_path, ratio=0.8):
  """Suffle the dataset and split the training set"""
  print("Splitting training and dev set ... ")

  if(dataset_name == "mscoco"): 
    with open(train_index_path) as fd:
      train_index = set([int(l[:-1]) for l in fd.readlines()])

    train, dev = [], []
    for i in range(len(train_sets)):
      if(i in train_index): train.append(train_sets[i])
      else: dev.append(train_sets[i])

  # if(dataset_name == "quora"): 
  #   train_index_file = "quora_train_index.txt"
  #   with open(train_index_file) as fd:
  #     train_index = set([int(l[:-1]) for l in fd.readlines()])

  #   dev_index_file = "quora_dev_index.txt"
  #   with open(dev_index_file) as fd:
  #     dev_index = set([int(l[:-1]) for l in fd.readlines()])

  #   train, dev = [], []
  #   for i in range(len(train_sets)):
  #     if(i in train_index): train.append(train_sets[i])
  #     elif(i in dev_index): dev.append(train_sets[i])

  print("Size of training set: %d" % len(train))
  print("Size of test set: %d" % len(dev))
  return train, dev

class DatasetMSCOCO(object):
  """The dataset class, read the raw data, process into intermediate 
  representation, and load the intermediate as batcher"""

  def __init__(self, config):
    """Initialize the dataset configuration"""
    self.dataset = config.dataset
    self.data_path = config.data_path[self.dataset]
    self.vocab_path = config.data_path[self.dataset]['vocab']
    self.close_key_path = config.data_path[self.dataset]['close_key']
    self.model_name = config.model_name
    self.max_sent_len = config.max_sent_len
    self.max_bow_len = 10 # hard code this 
    self.latent_vocab_size = config.latent_vocab_size

    # self.batch_size = config.batch_size

    self.stop_words = set(stopwords.words('english'))

    self.word2id = config.word2id
    self.id2word = config.id2word
    self.pad_id = config.pad_id
    self.start_id = config.start_id
    self.end_id = config.end_id
    self.unk_id = config.unk_id

    self._dataset = {"train": None, "dev": None, "test": None}
    self._ptr = {"train": 0, "dev": 0, "test": 0}
    self._reset_ptr = {"train": False, "dev": False, "test": False}
    return 

  @property
  def vocab_size(self): return len(self.word2id)

  def dataset_size(self, setname):
    return len(self._dataset[setname]['sentences'])
  
  def num_batches(self, setname, batch_size):
    setsize = self.dataset_size(setname)
    num_batches = setsize // batch_size + 1
    return num_batches

  def _update_ptr(self, setname, batch_size):
    if(self._reset_ptr[setname]):
      ptr = 0
      self._reset_ptr[setname] = False
    else: 
      ptr = self._ptr[setname]
      ptr += batch_size
      # truncate the last cases
      if(ptr + batch_size >= self.dataset_size(setname)):
        self._reset_ptr[setname] = True
    self._ptr[setname] = ptr
    return 

  def build(self):
    """Build the dataset to intermediate representation
    
    The data processing pipeline: 
    * read raw file 
    * calculate corpus statistics
    * split training and validation 
    * build vocabulary
    * normalize the text 
    """
    # read training sentences
    if(self.dataset == "mscoco"):
      train_sentences = mscoco_read_json(self.data_path["train"])
    elif(self.dataset == 'mscoco14'):
      train_sentences = mscoco_read_json(self.data_path["train"])
    elif(self.dataset == "quora"):
      train_sentences = quora_read(self.data_path["train"])

    # corpus_statistics(train_sentences)
    train_sentences, dev_sentences = train_dev_split(
      self.dataset, train_sentences, self.data_path['train_index']) 

    train_sentences_ = []
    for s in train_sentences: train_sentences_.extend(s)
    if(os.path.exists(self.vocab_path) == False):
      print('no given vocabulary, build one')
      self.word2id, self.id2word, _ = nlpp.build_vocab(
        train_sentences_, self.word2id, self.id2word)
      self.save_vocab()
    else:
      self.load_vocab()

    train_sentences_, train_lens_ = [],[]
    for s in train_sentences:
      s, sl = nlpp.normalize(s, self.word2id, self.max_sent_len)
      train_sentences_.append(s)
      train_lens_.append(sl)

    dev_sentences_, dev_lens_ = [], [] 
    for s in dev_sentences:
      s, sl = nlpp.normalize(s, self.word2id, self.max_sent_len)
      dev_sentences_.append(s)
      dev_lens_.append(sl)

    test_sentences = mscoco_read_json(self.data_path["test"])
    test_sentences_, test_lens_ = [], []
    for s in test_sentences:
      s, sl = nlpp.normalize(s, self.word2id, self.max_sent_len)
      test_sentences_.append(s)
      test_lens_.append(sl)

    self.stop_words = set([self.word2id[w] 
      if(w in self.word2id) else self.pad_id for w in self.stop_words])
    self.stop_words |= set(
      [self.start_id, self.end_id, self.unk_id, self.pad_id, 
      self.word2id['.'], self.word2id[',']])

    self._dataset = {'train': {'sentences': train_sentences_, 
                               'sent_lens': train_lens_ },
                     'dev':   {'sentences': dev_sentences_,
                               'sent_lens': dev_lens_ },
                     'test':  {'sentences': test_sentences_, 
                               'sent_lens': test_lens_ } }

    # self.close_key = None
    # if(os.path.exists(self.close_key_path) == False):
    #   self.build_close_key_index()
    # else:
    #   self.load_close_key_index()
    return 

  def save_vocab(self):
    print('saving vocabulary to %s' % self.vocab_path)
    with open(self.vocab_path, 'w') as fd:
      for i in self.id2word:
        fd.write('%d %s\n' % (i, self.id2word[i]))
    return 

  def load_vocab(self):
    print('loading vocabulary from %s' % self.vocab_path)
    with open(self.vocab_path) as fd:
      lines = fd.readlines()
      for l in lines:
        i, w = l.split()
        self.id2word[int(i)] = w
        self.word2id[w] = int(i)
    return 

  def next_batch(self, setname, batch_size):
    """Get next data batch
    
    Args:
      setname: a string, "train", "valid", or "test"
      batch_size: the size of the batch, an integer
    """
    # if(batch_size is None): batch_size = self.batch_size

    ptr = self._ptr[setname]
    sentences = self._dataset[setname]['sentences'][ptr: ptr + batch_size]
    sent_lens = self._dataset[setname]['sent_lens'][ptr: ptr + batch_size]

    if(setname == "train"):
      if(self.model_name.startswith('latent_temp_crf')):
        batch = self.build_batch_train_ltemp(sentences, sent_lens)
      elif(self.model_name == 'seq2seq'):
        batch = self.build_batch_train_seq2seq(sentences, sent_lens)
      else: 
        raise NotImplementedError('model %s not implemented' % self.model_name)
    else: # evaluation
      if(self.model_name.startswith('latent_temp_crf')):
        batch = self.build_batch_eval_ltemp(sentences, sent_lens)
      elif(self.model_name == 'seq2seq'):
        batch = self.build_batch_eval_seq2seq(sentences, sent_lens)
      else:
        raise NotImplementedError('model %s not implemented' % self.model_name)
    
    self._update_ptr(setname, batch_size)
    return batch

  def build_batch_train_seq2seq(self, sentences, sent_lens):
    inputs = []
    inp_lens = []
    targets = []
    for s, sl in zip(sentences, sent_lens):
      for i in range(len(s) - 1):
        si = s[i]
        sj = s[i + 1]
        si_l = sl[i]
        sj_l = sl[i + 1]
        inputs.append(si)
        inp_lens.append(si_l)
        targets.append(sj)

    batch = {'inputs': np.array(inputs),
             'inp_lens': np.array(inp_lens),
             'targets': np.array(targets)}
    return batch
  
  def build_batch_eval_seq2seq(self, sentences, sent_lens):
    sentences_ = []
    sent_lens_ = []
    references_ = []
    for s, sl in zip(sentences, sent_lens):
      sentences_.append(s[0])
      references_.append(s[1:])
      sent_lens_.append(sl[0])
    
    batch = { 'sentences': np.array(sentences_),
              'sent_lens': np.array(sent_lens_),
              'references': np.array(references_)}
    return batch 

  def build_batch_train_ltemp(self, sentences, sent_lens):
    """Build a training batch
    
    Args:
      sentences: normalized sentences, a list of paraphrase list. The second 
        level list is a list of sentences
      sent_lens: sentence length, a list of list. The second level list is a 
        list of integers
    """
    sentences_, sent_lens_ = [], []
    for i in range(len(sentences)):
      sentences_.extend(sentences[i])
      sent_lens_.extend(sent_lens[i])

    # get BOW from sentences
    bow = []
    mem_lens = []
    sentences_new = []
    sent_lens_new = []
    for s, sl in zip(sentences_, sent_lens_):
      b = set(s) - self.stop_words
      b, bl = nlpp.normalize([b], self.word2id, self.max_bow_len, False)
      if(bl[0] > 0):
        bow.extend(b)
        mem_lens.extend(bl)
        sentences_new.append(s)
        sent_lens_new.append(sl)

    batch = {
      'sentences': np.array(sentences_new), 
      'sent_lens': np.array(sent_lens_new),
      'keys': np.array(bow),
      'vals': np.array(bow),
      'mem_lens': np.array(mem_lens)}
    return batch

  def build_batch_eval_ltemp(self, sentences, sent_lens):
    """Build an evaluation batch
    
    Args:
      sentences: normalized sentences, a list of paraphrase list. The second 
        level list is a list of sentences
      sent_lens: sentence length, a list of list. The second level list is a 
        list of integers
    """
    bow = []
    mem_lens = []
    refs = []
    sents = []
    sent_lens_ = []
    for s, sl in zip(sentences, sent_lens):
      b = set(s[0]) - self.stop_words
      b, bl = nlpp.normalize([b], self.word2id, self.max_bow_len, False)
      if(bl[0] > 0):
        sents.append(s[0])
        refs.append(s[1:])
        bow.extend(b)
        mem_lens.extend(bl)
        sent_lens_.append(sl[0])
    batch = {'keys': np.array(bow), 
             'vals': np.array(bow),
             'mem_lens': np.array(mem_lens), 
             'sentences': np.array(sents),
             'sent_lens': np.array(sent_lens_),
             'references': np.array(refs)}
    return batch

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
    inspect_out = {}

    if('z_sample_ids' in inspect): 
      z_sample_ids = inspect['z_sample_ids']
      out += 'z_sample_ids\n'
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

    if(model_name.startswith('latent_temp_crf')):
      out += 's[0] tag sampled: \n'
      out += '          ' +\
        self.decode_sent_w_state(batch['sentences'][0], z_sample_ids[0]) + '\n'
      out += 's[0] sent:'
      out += self.decode_sent(batch['sentences'][0]) + '\n'
      out += '\n\n'

      out += 's[1] tag sampled: \n'
      out += '          ' +\
        self.decode_sent_w_state(batch['sentences'][1], z_sample_ids[1]) + '\n'
      out += 's[1] sent:'
      out += self.decode_sent(batch['sentences'][1]) + '\n'

      latent_vocab_stats = np.zeros(self.latent_vocab_size).astype(int)
      for i in range(self.latent_vocab_size):
        latent_vocab_stats[i] = int(np.sum(z_sample_ids == i))
      print('latent state stats:')
      print(latent_vocab_stats)


    if(do_not_print): return out
    else: print(out)
    return 

  def print_batch(self, batch, 
    out_dict=None, model_name=None, fd=None, fd_post=None, fd_full=None, 
    print_all=False):
    """Print out a test batch"""
    if('keys' in batch): range_key = 'keys'
    elif('inputs' in batch): range_key = 'inputs'
    elif('sentences' in batch): range_key = 'sentences'
    elif('sentence_set' in batch): range_key = 'sentence_set'
    else: raise NameError('check keys in batch!')

    if(print_all): 
      print_range = range(len(batch[range_key]))
    else: 
      if(fd is None): 
        print_range = np.random.choice(
          len(batch[range_key]), 5, replace=False)
      else: print_range = range(len(batch[range_key]))

    out = '' 
    pred_out = ''

    # if('inspect' in batch): 
    #   out += 'inspections:\n'
    #   out += self.print_inspect(batch['inspect'])

    for i in print_range:
      if('keys' in batch):
        out += 'mem:\n'
        for k, v in zip(batch['keys'][i][: batch['mem_lens'][i]], 
          batch['vals'][i][: batch['mem_lens'][i]]):
          out += self.id2word[k] + ' | '
        out += '\n'

      if('sentence_set' in batch):
        out += 'sentence_set:\n'
        for j in range(len(batch['sentence_set'][i])):
          out += '%d: ' % j
          out += self.decode_sent(batch['sentence_set'][i][j]) + '\n'

      if('references' in batch):
        out += 'references\n'
        for j in range(len(batch['references'][i])):
          out += '%d: ' % j
          out += self.decode_sent(batch['references'][i][j]) + '\n'

      if('inputs' in batch):
        out += 'inputs:\n'
        out += self.decode_sent(batch['inputs'][i]) + '\n'

      if('sentences' in batch):
        out += 'sentences:\n'
        out += self.decode_sent(batch['sentences'][i]) + '\n'

      if('targets' in batch):
        out += 'targets:\n'
        out += self.decode_sent(batch['targets'][i]) + '\n'

      if(out_dict is not None):
        out += 'predictions:\n'
        
        s_out = self.decode_sent_w_state(
          out_dict['predictions'][i], 
          out_dict['predictions_z'][i]) + '\n'
        out += ('%d:' % j) + s_out

        s_out = self.decode_sent(out_dict['predictions'][i], 
          add_eos=False) + '\n'
        pred_out += s_out

      out += '\n\n'

    out += '\n'
    if(fd is not None): fd.write(pred_out)
    else: print(out)

    if(fd_full is not None): fd_full.write(out + '\n\n')
    return 
