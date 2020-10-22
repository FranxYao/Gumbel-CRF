"""Testing code for CRF"""

import torch
# from modeling.structure.struct_linear_crf import LinearChainCRF
from modeling.structure.linear_crf import LinearChainCRF
from modeling import torch_model_utils as tmu
from config import Config

config = Config()
config.device = 'cpu'
config.latent_vocab_size = 2

print('hello world')

chain = LinearChainCRF(config)

seq_lens = torch.Tensor([2, 3]).type(torch.long).to(config.device)
emission_scores = torch.randn(2, 3, config.latent_vocab_size).to(config.device)

# test cases 
seq = torch.Tensor([[0, 0, 0], [0, 0, 0]]).long()
print(chain.log_prob(seq, emission_scores, seq_lens).exp())
seq = torch.Tensor([[0, 1, 0], [0, 0, 1]]).long()
print(chain.log_prob(seq, emission_scores, seq_lens).exp())
seq = torch.Tensor([[1, 0, 0], [0, 1, 0]]).long()
print(chain.log_prob(seq, emission_scores, seq_lens).exp())
seq = torch.Tensor([[1, 1, 0], [0, 1, 1]]).long()
print(chain.log_prob(seq, emission_scores, seq_lens).exp())
seq = torch.Tensor([[0, 0, 0], [1, 0, 0]]).long()
print(chain.log_prob(seq, emission_scores, seq_lens).exp())
seq = torch.Tensor([[0, 1, 0], [1, 0, 1]]).long()
print(chain.log_prob(seq, emission_scores, seq_lens).exp())
seq = torch.Tensor([[1, 0, 0], [1, 1, 0]]).long()
print(chain.log_prob(seq, emission_scores, seq_lens).exp())
seq = torch.Tensor([[1, 1, 0], [1, 1, 1]]).long()
print(chain.log_prob(seq, emission_scores, seq_lens).exp())

all_scores = chain.calculate_all_scores(emission_scores)
alpha, log_Z = chain.forward_score(emission_scores, seq_lens)
alpha_rev = tmu.reverse_sequence(alpha, seq_lens)

batch_size = emission_scores.size(0)
max_len = emission_scores.size(1)
num_class = emission_scores.size(2)
device = config.device
tau=1.0
sample_rev = torch.zeros(batch_size, max_len).type(torch.long).to(device)
all_scores_rev = tmu.reverse_sequence(all_scores, seq_lens)
relaxed_sample_rev = torch.zeros(batch_size, max_len, num_class)

w = alpha_rev[:, 0, :]
w -= log_Z.view(batch_size, -1)
print(0)
print(torch.exp(w)[0])
print(torch.exp(w)[0].sum())

relaxed_sample_rev[:, 0] = tmu.reparameterize_gumbel(w, tau)
sample_rev[:, 0] = relaxed_sample_rev[:, 0].argmax(dim=-1)

i = 1
y_after_to_current = all_scores_rev[:, i-1].transpose(1, 2)
w = tmu.batch_index_select(y_after_to_current, sample_rev[:, i-1])

sample, relaxed_sample = chain.rsample(emission_scores, seq_lens, 1.0)
print(sample)

emission_scores = torch.Tensor([[[2, 1], [3, 2], [5, 8]]]).log()
seq_lens = torch.Tensor([2]).type(torch.long).to(config.device)
alpha, log_Z = chain.forward_score(emission_scores, seq_lens)
chain.entropy(emission_scores, seq_lens)
# expect 25 * np.log(2) / 8 - 3 * np.log(3) / 4 = 1.3421