import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.nn.Parameter import Parameter


class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

def to_var(x, requires_grad=False, volatile=False):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, requires_grad=requires_grad, volatile=volatile)

# Custom FC layer for pruning
class MaskedLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super(MaskedLinear, self).__init__(in_features, out_features, bias)
        self.mask_flag = False
        self.mask = ''
        self.biasmask = torch.zeros(256)
        self.biassize = in_features
    
    def set_masks(self, mask, layername=''):
        
        self.mask = to_var(mask, requires_grad=False)
        self.weight.data = self.weight.data*self.mask.data
        if 'ff0' is in layername:
            self.biassize = 1024
            self.bias = Parameter(torch.zeros(1024).to('cuda'))
        else:
            self.bias = Parameter(torch.zeros(256).to('cuda'))
        self.mask_flag = True
    
    def get_masks(self):
        return self.mask
    
    def forward(self, x):
        if self.mask_flag == True:
            # applying pruning mask
            weight = self.weight*self.mask
            self.bias = Parameter(torch.zeros(self.biassize).to('cuda'))
            return F.linear(x, weight, self.bias)
        else:
            return F.linear(x, self.weight, self.bias)

class MaskedEmbedding(nn.Embedding):
    "Construct a layernorm module (See citation for details)."

    def __init__(self, length, model, padding_idx=None):
        super(MaskedEmbedding, self).__init__(length, model, padding_idx)
        self.mask_flag = False
        self.mask = ''

    def set_masks(self, mask, layername=''):
        self.mask = to_var(mask, requires_grad=False)
        self.weight.data = self.weight.data*self.mask.data
        self.mask_flag = True
    
    def get_masks(self):
        #print(self.mask_flag)
        return self.mask
    
    def forward(self, x):
        if self.mask_flag == True:
            # applying pruning mask
            weight = self.weight*self.mask
            return F.embedding(
                x, weight, self.padding_idx, self.max_norm,
                self.norm_type, self.scale_grad_by_freq, self.sparse)
        else:
            return F.embedding(
                x, self.weight, self.padding_idx, self.max_norm,
                self.norm_type, self.scale_grad_by_freq, self.sparse)
#class MaskedEmbedding(nn.Embedding):
