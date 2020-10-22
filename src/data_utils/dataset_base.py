class DatasetBase(object):
  def __init__(self):
    self._dataset = {"train": None, "dev": None, "test": None}
    self._ptr = {"train": 0, "dev": 0, "test": 0}
    self._reset_ptr = {"train": False, "dev": False, "test": False}
    return 

  @property
  def vocab_size(self): return len(self.word2id)

  @property
  def key_vocab_size(self): return len(self.key2id)

  def dataset_size(self, setname):
    raise NotImplementedError
    return 

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

  def num_batches(self, setname, batch_size):
    setsize = self.dataset_size(setname)
    num_batches = setsize // batch_size + 1
    return num_batches

  def build(self, config):
    return 

  def dataset_size(self, setname):
    return 

  def num_batches(self, setname, batch_size):
    return

  def next_batch(self, setname, batch_size):
    return 

  def print_batch(self, batch, fd):
    return 