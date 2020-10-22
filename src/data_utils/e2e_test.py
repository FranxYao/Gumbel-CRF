import numpy as np 

from collections import Counter
from config import Config
from data_utils.dataset_e2e import read_data, DatasetE2E, extend_vocab_with_keys, normalize_set
from data_utils.nlp_pipeline import build_vocab

trainset = read_data('../data/e2e-dataset/trainset.csv')

# how sparse the values are 
train_sents = [t[1] for t in trainset]
config = Config()
word2id, id2word = config.word2id, config.id2word
word2id, id2word, vocab = build_vocab(train_sents, word2id, id2word, vocab_size_threshold=1)
wset = set(word2id.keys())

max_tgt_len = 37
max_mem_len = 16

train_keys = []
for tb, _, _ in trainset:
  keys = [k for k, _ in tb]
  train_keys.extend(keys)
word2id, id2word, key2id, id2key =\
  extend_vocab_with_keys(word2id, id2word, train_keys)
(train_keys, train_vals, train_mem_lens, train_sentences, train_templates, 
      train_sent_lens) = normalize_set(
        trainset, word2id, max_tgt_len, max_mem_len)

train_lens = [len(s) for s in train_sents]
np.percentile(train_lens, [50, 90, 95, 99])
# >>
# array([21., 32., 35., 41.])

train_vals = []
for tb, _, _ in trainset:
  vals = [v for _, v in tb]
  train_vals.extend(vals)
train_vals = Counter(train_vals)
vset = set(train_vals.keys())
assert(len(vset - wset) == 0)

table_lens = [len(tb) for tb, _, _ in trainset]
np.percentile(table_lens, [50, 90, 95, 99])
# >>
# array([ 9., 13., 14., 16.])

config = Config()
config.max_sent_len = config.max_sent_len['e2e']
config.max_dec_len = config.max_dec_len['e2e']
config.max_bow_len = config.max_bow_len['e2e']
dset = DatasetE2E(config)
dset.build()
dset.dump_for_test()

dset.key_vocab_size
dset.vocab_size

setname = 'dev'
batch_size = 10
batch = dset.next_batch(setname, batch_size)
batch['sentences'] = batch['templates']
dset.print_batch(batch)
num_batches = dset.num_batches(setname, batch_size)
for _ in range(num_batches):
  batch = dset.next_batch(setname, batch_size)
  for k in batch: 
    if(np.isnan(np.sum(batch[k]))):
      print(batch)
  
dset.print_batch(batch)
