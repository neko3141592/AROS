import torch
import torch.nn as nn
import numpy as np

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.depth = d_model // num_heads

        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        self.dense = nn.Linear(d_model, d_model)

    def split_heads(self, x, batch_size):
        x = x.view(batch_size, -1, self.num_heads, self.depth)
        return x.permute(0, 2, 1, 3)

    def forward(self, x):
        batch_size = x.size(0)
        q = self.split_heads(self.wq(x), batch_size)
        k = self.split_heads(self.wk(x), batch_size)
        v = self.split_heads(self.wv(x), batch_size)

        # Scaled dot-product attention
        scaled_attention = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.depth)
        attention_weights = nn.functional.softmax(scaled_attention, dim=-1)
        output = torch.matmul(attention_weights, v)

        output = output.permute(0, 2, 1, 3).contiguous()
        output = output.view(batch_size, -1, self.d_model)
        return self.dense(output)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class Transformer(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, d_ff, input_vocab_size, target_vocab_size):
        super(Transformer, self).__init__()
        self.encoder = nn.Embedding(input_vocab_size, d_model)
        self.decoder = nn.Embedding(target_vocab_size, d_model)
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.positional_encoding = PositionalEncoding(d_model)

    def forward(self, src, tgt):
        src = self.positional_encoding(self.encoder(src))
        tgt = self.positional_encoding(self.decoder(tgt))
        output = self.mha(src)
        return output

# Example usage
if __name__ == '__main__':
    transformer = Transformer(num_layers=6, d_model=512, num_heads=8, d_ff=2048, input_vocab_size=10000, target_vocab_size=10000)
    src = torch.randint(0, 10000, (32, 10))  # (batch_size, sequence_length)
    tgt = torch.randint(0, 10000, (32, 10))
    output = transformer(src, tgt)
    print(output.shape)  # Should output (32, 10, 512)
