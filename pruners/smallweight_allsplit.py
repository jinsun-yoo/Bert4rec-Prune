#%%writefile pruners/smallweight_allsplit.py
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
        all_position = []
        all_feed = []
        for name, p in model.named_parameters():
            if len(p.data.size()) != 1:
                if 'linear' in name:
                    all_weights += list(p.cpu().data.abs().numpy().flatten())
                elif 'token' in name:
                    all_embed += list(p.cpu().data.abs().numpy().flatten())
                elif 'position' in name:
                    all_position += list(p.cpu().data.abs().numpy().flatten())
                elif 'feed' in name:
                    all_feed += list(p.cpu().data.abs().numpy().flatten())
        threshold = np.percentile(np.array(all_weights), pruning_perc) # For example, median = np.percnetile(some_vector, 50.)
        threshold_embedded = np.percentile(np.array(all_embed), pruning_perc_embed) # For example, median = np.percnetile(some_vector, 50.)
        threshold_feed = np.percentile(np.array(all_feed), pruning_perc_feed) # For example, median = np.percnetile(some_vector, 50.)
        threshold_position = np.percentile(np.array(all_position), pruning_perc_feed) # For example, median = np.percnetile(some_vector, 50.)

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
                elif 'position' in name:
                    pruned_inds = p.data.abs() > threshold_position
                    masks.append(pruned_inds.float())
                elif 'feed' in name:
                    pruned_inds = p.data.abs() > threshold_feed
                    masks.append(pruned_inds.float())
                    
        return masks