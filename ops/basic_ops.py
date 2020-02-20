import torch


class Identity(torch.nn.Module):
    def forward(self, input):
        return input


class SegmentConsensus(torch.autograd.Function):

    def __init__(self, consensus_type, dim=1):
        self.consensus_type = consensus_type
        self.dim = dim
        self.shape = None

    def forward(self, input_tensor):
        self.shape = input_tensor.size()
        if self.consensus_type == 'avg':
            output = input_tensor.mean(dim=self.dim, keepdim=True)
        elif self.consensus_type == 'identity':
            output = input_tensor
        else:
            output = None

        return output

    def backward(self, grad_output):
        if self.consensus_type == 'avg':
            grad_in = grad_output.expand(self.shape) / float(self.shape[self.dim])
        elif self.consensus_type == 'identity':
            grad_in = grad_output
        else:
            grad_in = None

        return grad_in


class ConsensusModule(torch.nn.Module):

    def __init__(self, consensus_type, dim=1, args=None):
        super(ConsensusModule, self).__init__()
        self.consensus_type = consensus_type if consensus_type != 'rnn' else 'identity'
        self.dim = dim
        self.args=args

    def forward(self, input):
        if self.consensus_type=="scsampler": # TODO(yue)
            res = []
            ind = torch.topk(torch.max(input.detach(), dim=2).values, dim=1, k=self.args.top_k).indices
            for bi in range(input.shape[0]):
                res.append(torch.stack([input[bi, k, :] for k in ind[bi]]))
            output = torch.mean(torch.stack(res), dim=1, keepdim=True)
            return output
        else:
            return SegmentConsensus(self.consensus_type, self.dim)(input)




# class ConsensusModule(torch.nn.Module):
#
#     def __init__(self, consensus_type, dim=1):
#         super(ConsensusModule, self).__init__()
#         self.consensus_type = consensus_type if consensus_type != 'rnn' else 'identity'
#         self.dim = dim
#
#     def forward(self, input):
#         return input.mean(dim=self.dim, keepdim=True)