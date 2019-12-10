#%%writefile pruners/smallweight_embedsplit.py
from .base import AbstractPruner
from models.bert_modules.custom_layers import *
import torch.nn as nn
import numpy as np


class Pruner_Linear_Embed(AbstractPruner):
    def __init__(self, args, model):
        super().__init__(args, model)

    @classmethod
    def code(cls):
        return 'pruner_linear_embed'

    def weight_prune(self, model, pruning_perc, pruning_perc_embed):
        '''
        Prune pruning_perc% weights globally (not layer-wise)
        arXiv: 1606.09274
        '''    
        all_weights = []
        all_embed = []
        for name, p in model.named_parameters():
            if len(p.data.size()) != 1:
                if 'linear' in name:
                    all_weights += list(p.cpu().data.abs().numpy().flatten())
                elif 'token' in name:
                    all_embed += list(p.cpu().data.abs().numpy().flatten())
        threshold = np.percentile(np.array(all_weights), pruning_perc) # For example, median = np.percnetile(some_vector, 50.)
        threshold_embedded = np.percentile(np.array(all_embed), pruning_perc_embed) # For example, median = np.percnetile(some_vector, 50.)

        # generate mask
        masks = []
        for name, p in model.named_parameters():
            if len(p.data.size()) != 1:
                if 'linear' in name:
                    pruned_inds = p.data.abs() > threshold
                    masks.append(pruned_inds.float())
                elif 'token' in name:
                    pruned_inds = p.data.abs() > threshold_embedded
                    masks.append(pruned_inds.float())
                    
        return masks