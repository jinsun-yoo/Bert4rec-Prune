from .base import AbstractPruner

import torch.nn as nn
import numpy as np


class SmallWeightPruner(AbstractPruner):
    def __init__(self, args, model):
        super().__init__(args, model)

    @classmethod
    def code(cls):
        return 'smallweight'

    def weight_prune(self, model, pruning_perc, pruning_perc_embed):
        '''
        Prune pruning_perc% weights globally (not layer-wise)
        arXiv: 1606.09274
        '''    
        all_weights = []
        for name, p in model.named_parameters():
            if len(p.data.size()) != 1:
                #print(f'adding weigths of {name}')
                #print(f'original data is {p.data}')
                #print(f'list is {(p.cpu().data.abs().numpy().flatten)}')
                all_weights += list(p.cpu().data.abs().numpy().flatten())
        threshold = np.percentile(np.array(all_weights), pruning_perc) # For example, median = np.percnetile(some_vector, 50.)

        # generate mask
        masks = []
        for name, p in model.named_parameters():
            if len(p.data.size()) != 1:
                pruned_inds = p.data.abs() > threshold
                masks.append(pruned_inds.float())
        return masks