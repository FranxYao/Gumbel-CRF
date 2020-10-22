class FTModel(object):
  """Fast research-level engineering with Torch, base model"""

  def __init__(self):
    self.model = None # to be instantialized with a torch model 
    return 

  def train(self):
    self.model.train()
    return 

  def eval(self):
    self.model.eval()
    return 

  def to(self, device):
    self.model.to(device)
    return 

  def parameters(self):
    return self.model.parameters()

  def named_parameters(self):
    return self.model.named_parameters()

  def state_dict(self):
    return self.model.state_dict()

  def load_state_dict(self, ckpt):
    self.model.load_state_dict(ckpt)
    return 

  def zero_grad(self):
    self.model.zero_grad()
    return 

  def train_step(self):
    raise NotImplementedError

  def infer_step(self):
    raise NotImplementedError