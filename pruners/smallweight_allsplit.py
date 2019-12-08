#%%writefile pruners/smallweight_embedsplit.py
from .base import AbstractPruner
from models.bert_modules.custom_layers import *
import torch.nn as nn
import numpy as np


class SmallWeightSplitAll(AbstractPruner):
    def __init__(self, args, model):
        super().__init__(args, model)

    @classmethod
    def code(cls):
        return 'smallweightsplitall'

    def weight_prune(self, model, pruning_perc, pruning_perc_embed = 0, pruning_perc_feed = 0):
        '''
        Prune pruning_perc% weights globally (not layer-wise)
        arXiv: 1606.09274
        '''    
        all_weights = []
        all_embed = []
        all_feed = []
        for name, p in model.named_parameters():
            if len(p.data.size()) != 1:
                if 'linear' in name:
                    all_weights += list(p.cpu().data.abs().numpy().flatten())
                elif 'token' in name:
                    all_embed += list(p.cpu().data.abs().numpy().flatten())
                elif 'feed' in name:
                    all_feed += list(p.cpu().data.abas().numpy().flatten())
        threshold = np.percentile(np.array(all_weights), pruning_perc) # For example, median = np.percnetile(some_vector, 50.)
        threshold_embedded = np.percentile(np.array(all_embed), pruning_perc_embed) # For example, median = np.percnetile(some_vector, 50.)
        threshold_feed = np.percentile(np.array(all_embed), pruning_perc_feed) # For example, median = np.percnetile(some_vector, 50.)
        """
                #print(f'adding weigths of {name}')
                #print(f'original data is {p.data}')
                #print(f'list is {(p.cpu().data.abs().numpy().flatten)}')
                all_weights += list(p.cpu().data.abs().numpy().flatten())
        threshold = np.percentile(np.array(all_weights), pruning_perc) # For example, median = np.percnetile(some_vector, 50.)

        all_embed = []
        for name, p in model.named_parameters():
            print(type(p))
            if len(p.data.size()) != 1 and type(p) == MaskedEmbedded:
                #print(f'adding weigths of {name}')
                #print(f'original data is {p.data}')
                #print(f'list is {(p.cpu().data.abs().numpy().flatten)}')
                all_embed += list(p.cpu().data.abs().numpy().flatten())
        threshold_embedded = np.percentile(np.array(all_weights), pruning_perc_embed) # For example, median = np.percnetile(some_vector, 50.)
        """
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
                elif 'feed' in name:
                    pruned_inds = p.data.abs() > threshold_embedded
                    masks.append(pruned_inds.float())
                    
        return masks