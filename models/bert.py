from .base import BaseModel
from .bert_modules.bert import BERT

import torch.nn as nn


class BERTModel(BaseModel):
    def __init__(self, args):
        super().__init__(args)
        self.bert = BERT(args)
        self.out = nn.Linear(self.bert.hidden, args.bert_num_items + 1)

    @classmethod
    def code(cls):
        return 'bert'

    def forward(self, x):
        x = self.bert(x)
        return self.out(x)
    
    def set_masks(self, masks):
        for mask in masks:
            print(mask)
            print(mask.data.size())
        self.bert.transformer_blocks[0].attention.linear_layers[0].set_masks((masks[2]))
        self.bert.transformer_blocks[0].attention.linear_layers[1].set_masks((masks[3]))
        self.bert.transformer_blocks[0].attention.linear_layers[2].set_masks((masks[4]))
        self.bert.transformer_blocks[0].attention.output_linear.set_masks((masks[5]))
        #self.bert.transformer_blocks[0].feed_forward.w_1.set_masks((masks[6]))
        #self.bert.transformer_blocks[0].feed_forward.w_2.set_masks((masks[7]))
        #self.bert.transformer_blocks[0].input_sublayer.norm.set_masks((masks[8]))
        #self.bert.transformer_blocks[0].input_sublayer.norm.set_masks((masks[9]))
        self.bert.transformer_blocks[0].attention.linear_layers[0].set_masks((masks[10]))
        self.bert.transformer_blocks[0].attention.linear_layers[1].set_masks((masks[11]))
        self.bert.transformer_blocks[0].attention.linear_layers[2].set_masks((masks[12]))
        self.bert.transformer_blocks[0].attention.output_linear.set_masks((masks[13]))
        #self.bert.transformer_blocks[0].feed_forward.w_1.set_masks((masks[14]))
        #self.bert.transformer_blocks[0].feed_forward.w_2.set_masks((masks[15]))
        #self.bert.transformer_blocks[0].input_sublayer.norm.set_masks((masks[16]))
        #self.bert.transformer_blocks[0].input_sublayer.norm.set_masks((masks[17]))