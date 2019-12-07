import torch.nn as nn
from models.bert_modules.custom_layers import MaskedEmbedding

class TokenEmbedding(MaskedEmbedding):
    def __init__(self, vocab_size, embed_size=512):
        super().__init__(vocab_size, embed_size, padding_idx=0)
