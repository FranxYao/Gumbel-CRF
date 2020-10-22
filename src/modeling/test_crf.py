from modeling.structure.linear_crf import LinearChainCRF
import torch 
from modeling import torch_model_utils as tmu 
from time import time

device = 'cpu'

class Config:
  latent_vocab_size = 3
  device=device 

config = Config()
chain = LinearChainCRF(config)

emission_scores = torch.randn(2, 5, config.latent_vocab_size).to(device)
seq_lens = torch.tensor([3, 2]).long()

tau = 1.0
sample, relaxed_sample, _, sample_log_prob_stepwise = chain.rsample(
  emission_scores, seq_lens, tau, return_prob=True)

trainsition = chain.transition
all_scores = chain.calculate_all_scores(emission_scores)
all_scores_rev = tmu.reverse_sequence(all_scores, seq_lens)
alpha, log_Z = chain.forward_score(emission_scores, seq_lens)
beta = chain.backward_score(emission_scores, seq_lens)
for i in range(5):
  print(torch.logsumexp(alpha[0][i] + beta[0][i], 0))

for i in range(5):
  print(torch.logsumexp(alpha[1][i] + beta[1][i], 0))

seq = torch.Tensor([[0, 1, 2, 1, 0], [1, 1, 2, 2, 0]]).long()
log_marginal = chain.marginal(seq, emission_scores, seq_lens)