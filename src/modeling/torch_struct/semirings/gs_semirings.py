import torch
import torch.distributions
from .semirings import _BaseLog

def GumbelSoftmaxSemiring(temp):
    class _GumbelLogSumExp(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input, dim):
            ctx.save_for_backward(input, torch.tensor(dim))
            return torch.logsumexp(input, dim=dim)

        @staticmethod
        def backward(ctx, grad_output):
            logits, dim = ctx.saved_tensors
            grad_input = None
            if ctx.needs_input_grad[0]:
                def sample(ls):
                    update = (ls + torch.distributions.Gumbel(0, 1).sample((ls.shape[-1],)).to(ls.device)) / temp
                    return update.softmax(-1)
                if dim == -1:
                    s = sample(logits)
                else:
                    dim = dim if dim >= 0 else logits.dim() + dim
                    perm = [i for i in range(logits.dim()) if i != dim] + [dim]
                    rev_perm = [a for a, b in sorted(enumerate(perm), key=lambda a: a[1])]
                    s = sample(logits.permute(perm)).permute(rev_perm)
                grad_input = grad_output.unsqueeze(dim).mul(s)
            return grad_input, None
            
    class _GumbelSoftmaxSemiring(_BaseLog):
        @staticmethod
        def sum(xs, dim=-1):
            return _GumbelLogSumExp.apply(xs, dim)

    return _GumbelSoftmaxSemiring
