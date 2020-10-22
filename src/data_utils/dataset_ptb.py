
import numpy as np 
from tqdm import tqdm 
from .dataset_e2e import DatasetE2E
from . import nlp_pipeline as nlpp

def read_vocab(vocab_path, word2id, id2word):
  with open(vocab_path) as fd:
    lines = fd.readlines()
  wid = len(word2id)
  for l in lines:
    w = l[:-1]
    word2id[w] = wid
    id2word[wid] = w
    wid += 1
  return word2id, id2word

def read_data(dpath):
  with open(dpath) as fd:
    lines = fd.readlines()
  sentences = [l.split() for l in lines]
  return sentences

class DatasetPTB(DatasetE2E):

  def __init__(self, config):
    super(DatasetPTB, self).__init__(config)
    self.data_path = config.data_path['ptb']

    self.word2id = config.word2id
    self.id2word = config.id2word

    self.max_sent_len = config.max_sent_len
    self.max_bow_len = config.max_bow_len

    self._dataset = {'train': None, 'dev': None, 'test': None}
    self._ptr = {"train": 0, "dev": 0, "test": 0}
    return 

  @property
  def vocab_size(self): return len(self.word2id)

  def dataset_size(self, setname):
    return len(self._dataset[setname]['sentences'])

  def num_batches(self, setname, batch_size):
    setsize = self.dataset_size(setname)
    num_batches = setsize // batch_size + 1
    return num_batches

  def build(self):
    max_sent_len = self.max_sent_len
    self.word2id, self.id2word = read_vocab(
      self.data_path['vocab'], self.word2id, self.id2word)

    train = read_data(self.data_path['train'])
    dev = read_data(self.data_path['dev'])
    test = read_data(self.data_path['test'])
    print('number of sentences, %d train, %d dev, %d test' % 
     (len(train), len(dev), len(test)))

    train_sents, train_lens = nlpp.normalize(train, 
      self.word2id, max_sent_len)
    train_bow = [nlpp.sent_to_bow(s, self.max_bow_len) for s in train_sents]
    dev_sents, dev_lens = nlpp.normalize(dev, 
      self.word2id, max_sent_len)
    dev_bow = [nlpp.sent_to_bow(s, self.max_bow_len) for s in dev_sents]
    test_sents, test_lens = nlpp.normalize(test, 
      self.word2id, max_sent_len)
    test_bow = [nlpp.sent_to_bow(s, self.max_bow_len) for s in test_sents]
    print('number of sentences, %d train, %d dev, %d test' % 
      (len(train_sents), len(dev_sents), len(test_sents)))
    self._dataset = {'train': {'sentences': train_sents,
                               'sent_bow': train_bow, 
                               'sent_lens': train_lens},
                     'dev':   {'sentences': dev_sents,
                               'sent_bow': dev_bow,
                               'sent_lens': dev_lens},
                     'test':  {'sentences': test_sents,
                               'sent_bow': test_bow,
                               'sent_lens': test_lens}}
    return 

  def _update_ptr(self, setname, batch_size):
    ptr = self._ptr[setname]
    ptr += batch_size
    if(ptr == self.dataset_size(setname)): 
      ptr = 0
    if(ptr + batch_size > self.dataset_size(setname)):
      ptr = self.dataset_size(setname) - batch_size
    self._ptr[setname] = ptr
    return 

  def next_batch(self, setname, batch_size):
    ptr = self._ptr[setname]

    sentences = self._dataset[setname]['sentences'][ptr: ptr + batch_size]
    sent_bow = self._dataset[setname]['sent_bow'][ptr: ptr + batch_size]
    sent_lens = self._dataset[setname]['sent_lens'][ptr: ptr + batch_size]

    batch = {'sentences': np.array(sentences),
             'sent_bow': np.array(sent_bow),
             'sent_lens': np.array(sent_lens) # sent_len + _GOO
             }

    self._update_ptr(setname, batch_size)
    return batch

  def print_inspect(self, inspect, batch, model_name, do_not_print=False):
    out = ''
    z_sample_ids = inspect['z_sample_ids']
    batch['sent_lens'] += 1

    if(model_name == 'latent_temp_seq_label_vae'):
      out += 's[0] tag sampled: \n'
      out += '          ' +\
        self.decode_sent_w_state(batch['sentences'][0], z_sample_ids[0]) + '\n'
      out += 's[1] tag sampled: \n'
      out += '          ' +\
        self.decode_sent_w_state(batch['sentences'][1], z_sample_ids[1]) + '\n'
      out += 's[2] tag sampled: \n'
      out += '          ' +\
        self.decode_sent_w_state(batch['sentences'][2], z_sample_ids[2]) + '\n'

    out += 'state_length_statistics:\n'
    stats = self.state_length_statistics(z_sample_ids, batch['sent_lens'])
    out += ', '.join('%d - %d' % (l, stats[l]) for l in stats) + '\n'
    out += 'length 2/1 = %.4f, 3/1 = %.4f, >3/1 = %.4f' %\
      (stats[2] / stats[1], stats[3] / stats[1],\
        (stats[0] - stats[1] - stats[2] - stats[3]) / stats[1]) + '\n'
    
    out += 'latent_state_vocab_ent:\n'
    latent_state_vocab_ent = inspect['latent_state_vocab_ent']
    out += ''.join(
      '%d-%.4g  ' % (si, ent) for si, ent\
        in enumerate(latent_state_vocab_ent[:20]))
    out += '\n'

    if(do_not_print): return out
    else: print(out)
    return out

  def print_batch(self, batch, fd):
    return 