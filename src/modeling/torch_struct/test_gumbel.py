import torch 
import torch_struct 

batch, N, C = 3, 7, 2
log_potentials = torch.rand(batch, N, C, C)
dist = torch_struct.LinearChainCRF(log_potentials)

sample = dist.sample([1])
rsample = dist.rsample([1], 0.001)
print(rsample[0, 0].sum(-1))