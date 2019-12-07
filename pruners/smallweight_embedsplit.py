from .base import AbstractPruner
from models.bert_modules.custom_layers import *
import torch.nn as nn
import numpy as np


class SmallWeightSplitEmbeddedPruner(AbstractPruner):
    def __init__(self, args, model):
        super().__init__(args, model)

    @classmethod
    def code(cls):
        return 'smallweightsplitembeddedpruner'

    def weight_prune(self, model, pruning_perc, pruning_perc_embed):
        '''
        Prune pruning_perc% weights globally (not layer-wise)
        arXiv: 1606.09274
        '''    
        all_weights = []
        for name, p in model.named_parameters():
            if len(p.data.size()) != 1 and type(p) is MaskedLinear:
                #print(f'adding weigths of {name}')
                #print(f'original data is {p.data}')
                #print(f'list is {(p.cpu().data.abs().numpy().flatten)}')
                all_weights += list(p.cpu().data.abs().numpy().flatten())
        threshold = np.percentile(np.array(all_weights), pruning_perc) # For example, median = np.percnetile(some_vector, 50.)

        all_embed = []
        for name, p in model.named_parameters():
            print(type(p))
            if len(p.data.size()) != 1 and type(p) is MaskedEmbedded:
                #print(f'adding weigths of {name}')
                #print(f'original data is {p.data}')
                #print(f'list is {(p.cpu().data.abs().numpy().flatten)}')
                all_weights += list(p.cpu().data.abs().numpy().flatten())
        threshold_embedded = np.percentile(np.array(all_weights), pruning_perc_embed) # For example, median = np.percnetile(some_vector, 50.)

        # generate mask
        masks = []
        for name, p in model.named_parameters():
            if len(p.data.size()) != 1:
                if type(p) is MaskedLinear:
                    pruned_inds = p.data.abs() > threshold
                    masks.append(pruned_inds.float())
                elif type(p) is MaskedEmbedded:
                    pruned_inds = p.data.abs() > threshold_embedded
                    masks.append(pruned_inds.float())
                else:
                    print(name)
                    print(type(p))
        return masks