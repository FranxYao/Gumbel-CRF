
import numpy as np 

import nltk 
from nltk.corpus import stopwords
from collections import Counter

def normalize(sentences, word2id, max_sent_len, add_start_end=True):
  """Normalize the sentences by the following procedure
  - word to index, if already given index, skip this step 
  - add unk
  - pad/ cut the sentence length
  - record the sentence length

  Args: 
    sentences: a list of sentences, sentences are tokenized, i.e. a sentence 
      is a list of words, and a word is a string
    word2id: word index, a dictionary
    max_sent_len: maximum sentence length, a integer

  Returns: 
    sent_sets: a list of normalized sentences 
      A sentence is a list of word index
    sent_len_sets: a list of sentence length, 
      length = true length + 1 (_GOO or _EOS)
  """
  sent_sets = []
  sent_len_sets = []
  max_sent_len = max_sent_len + 1

  for s in sentences:
    if(add_start_end):
      s_ = [word2id["_GOO"]]
    else: s_ = []
    for w in s:
      if(isinstance(w, int)): s_.append(w) # if already index, skip 
      else:
        if(w in word2id): s_.append(word2id[w])
        else: s_.append(word2id["_UNK"])
    if(add_start_end):
      s_.append(word2id["_EOS"])
    
    s_ = s_[: max_sent_len]
    if(len(s_) < max_sent_len):
      s_len = len(s_)
      if(add_start_end): s_len = s_len - 1
      for i in range(max_sent_len - len(s_)): s_.append(word2id["_PAD"])
    else: 
      if(add_start_end): 
        s_[-1] = word2id["_EOS"]
        s_len = max_sent_len - 1
      else: 
        s_len = max_sent_len
    
    sent_sets.append(s_)
    sent_len_sets.append(s_len)
  return sent_sets, sent_len_sets

def sent_to_bow(s, max_bow, pad_id=0):
  """Sentence to BOW

  Args:
    s: type=list(int) a list of word index
    max_bow: type=int
    pad_id: type=int

  Returns:
    out: type=list(int), padded to max_bow length with pad_id
  """
  s = set(s)
  out = [w for w in s]
  out_len = len(out)
  if(out_len < max_bow): 
    for _ in range(max_bow - out_len): out.append(pad_id)
  else: out = out[: max_bow]
  return out

def build_vocab(training_set, 
  word2id=None, id2word=None, vocab_size_threshold=5):
  """Get the vocabulary from the training set"""
  vocab = []
  for s in training_set:
    vocab.extend(s)

  vocab = Counter(vocab)
  vocab_truncate = [w for w in vocab if vocab[w] >= vocab_size_threshold]

  if(word2id is None):
    word2id = {}
    id2word = {}
  i = len(word2id)
  for w in vocab_truncate:
    word2id[w] = i
    id2word[i] = w
    i += 1
  
  assert(len(word2id) == len(id2word))
  print("vocabulary size: %d" % len(word2id))
  return word2id, id2word, vocab

def extend_vocab_with_keys(word2id, id2word, keys):
  """join key vocab and the word vocab, but keep a seperate key dict"""
  key2id, id2key = {}, {}
  i = len(word2id)
  keys = set(keys)
  for k in keys:
    key2id[k] = i
    id2key[i] = k
    word2id[k] = i
    id2word[i] = k
    i += 1
  return word2id, id2word, key2id, id2key