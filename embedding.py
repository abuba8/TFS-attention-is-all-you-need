import torch
import torch.nn as nn
import math

class InputEmbeddings(nn.Module):
    def __init__(self, d_modal, vocab_size):
        super().__init__()
        self.d_model = d_modal
        self.vocab_size = vocab_size
        self.embeddings = nn.Embedding(vocab_size, d_modal)

    def forward(self, x):
        # In paper, the embedding layer, they've multiplied the weights by sqrt of d_model
        return self.embeddings(x) * math.sqrt(self.d_model) 
    


class PositionEmbedding(nn.Module):
    def __init__(self, d_model, seq_len, dropout):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = dropout
        pe = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0)/d_model))

        pe[:, 0::2] = torch.sin(position*div_term)
        pe[:, 1::2] = torch.cos(position*div_term)
        pe = pe.unsqueeze(0) # (1, seq_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad(False)
        return self.dropout(x)