
import torch 

import torch.nn as nn
import torch.nn.init as init
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class LSTMEncoder(nn.Module):
  def __init__(self, config):
    super(LSTMEncoder, self).__init__()

    self.lstm_bidirectional = config.lstm_bidirectional

    # self.reverse_input = config.reverse_input
    self.state_size = config.state_size
    if(config.lstm_layers == 1): dropout = 0.
    else: dropout = config.dropout
    self.cell = nn.LSTM(config.embedding_size, 
                        self.state_size, 
                        num_layers=config.lstm_layers, 
                        bidirectional=config.lstm_bidirectional, 
                        dropout=dropout)
    self.device = config.device

    self.bridge_h = nn.Linear(self.state_size * 2, self.state_size)
    self.bridge_c = nn.Linear(self.state_size * 2, self.state_size)
    self.dropout = nn.Dropout(config.dropout)

    # Bidirectional learned initial state
    if(config.lstm_bidirectional):
      self.init_hidden_h = torch.nn.Parameter(
        init.normal_(torch.zeros(
          2 * config.lstm_layers, 1, self.state_size, 
          requires_grad=True, device=self.device), mean=0, std=0.01))
      self.init_hidden_c = torch.nn.Parameter(
        init.normal_(torch.zeros(
          2 * config.lstm_layers, 1, self.state_size, 
          requires_grad=True, device=self.device), mean=0, std=0.01))
    else: 
      self.init_hidden_h = torch.nn.Parameter(
        init.normal_(torch.zeros(
          config.lstm_layers, 1, self.state_size, 
          requires_grad=True, device=self.device), mean=0, std=0.01))
      self.init_hidden_c = torch.nn.Parameter(
        init.normal_(torch.zeros(
          config.lstm_layers, 1, self.state_size, 
          requires_grad=True, device=self.device), mean=0, std=0.01))

    self.init_hidden = (self.init_hidden_h, self.init_hidden_h)
    return 

  def init_state(self, batch_size):
    state = (self.init_hidden[0].repeat(1, batch_size, 1), 
      self.init_hidden[1].repeat(1, batch_size, 1))
    return state

  def forward(self, enc_inputs, enc_lens):
    """
    Args:
      enc_inputs: type = torch.tensor(Float)
                  shape = [batch, max_len, embedding_size]
      enc_lens: type = torch.tensor(Int)
                shape = [batch]

    Returns:
      enc_outputs: type = torch.tensor(Float)
                   shape = [batch, max(enc_lens), state_size]
        NOTE: the output length is NOT the input length. i.e:
        enc_inputs.size(1) != enc_outputs.size(1)
      enc_state = (h, c), type = tuple 
        h: type = torch.tensor(Float), 
           shape = [num_layers, batch, hidden_state]
        c: type = torch.tensor(Float), 
           shape = [num_layers, batch, hidden_state]
    """
    batch_size = enc_inputs.shape[0]
    enc_init_state = self.init_state(batch_size)

    enc_inputs = self.dropout(enc_inputs)
    enc_inputs = pack_padded_sequence(enc_inputs, enc_lens, batch_first=True,
      enforce_sorted=False)
    enc_outputs, enc_state = self.cell(enc_inputs, enc_init_state)
    enc_outputs, _ = pad_packed_sequence(enc_outputs, batch_first=True) 

    h = enc_state[0]
    c = enc_state[1]
    if(self.lstm_bidirectional):
      max_len = enc_outputs.size(1)
      enc_outputs = enc_outputs.view(batch_size, max_len, 2, self.state_size)
      enc_outputs = enc_outputs.sum(dim=2)
      
      h = torch.cat([h[0: h.size(0): 2], h[1: h.size(0): 2]], 2)
      c = torch.cat([c[0: c.size(0):2], c[1: c.size(0):2]], 2)
      h = self.bridge_h(h)
      c = self.bridge_c(c)

    enc_outputs = self.dropout(enc_outputs)
    enc_state = (h, c)
    return enc_outputs, enc_state