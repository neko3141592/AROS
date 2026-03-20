import torch
import torch.nn as nn
import numpy as np

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, h):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.h = h
        self.d_k = d_model // h
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.fc = nn.Linear(d_model, d_model)

    def forward(self, q, k, v):
        batch_size = q.size(0)
        q = self.W_q(q).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        k = self.W_k(k).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        v = self.W_v(v).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.d_k)
        attn = torch.softmax(scores, dim=-1)
        output = torch.matmul(attn, v)
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        return self.fc(output)

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        return self.linear2(self.dropout(torch.relu(self.linear1(x))))

class EncoderLayer(nn.Module):
    def __init__(self, d_model, h, d_ff):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, h)
        self.ffn = FeedForward(d_model, d_ff)
        self.layernorm1 = nn.LayerNorm(d_model)
        self.layernorm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        attn_output = self.self_attn(x, x, x)
        x = self.layernorm1(x + attn_output)
        ffn_output = self.ffn(x)
        return self.layernorm2(x + ffn_output)

class Encoder(nn.Module):
    def __init__(self, N, d_model, h, d_ff):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList([EncoderLayer(d_model, h, d_ff) for _ in range(N)])
        self.positional_encoding = PositionalEncoding(d_model)

    def forward(self, x):
        x = self.positional_encoding(x)
        for layer in self.layers:
            x = layer(x)
        return x

class Transformer(nn.Module):
    def __init__(self, N, d_model, h, d_ff):
        super(Transformer, self).__init__()
        self.encoder = Encoder(N, d_model, h, d_ff)

    def forward(self, x):
        return self.encoder(x)

if __name__ == '__main__':
    N = 6
    d_model = 512
    d_ff = 2048
    h = 8
    model = Transformer(N, d_model, h, d_ff)
    sample_input = torch.rand(10, 20, d_model)  # (batch_size, seq_len, d_model)
    output = model(sample_input)
    print(output.shape)