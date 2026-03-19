import torch
import torch.nn as nn
import torch.optim as optim

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(MultiHeadAttention, self).__init__()
        self.heads = heads
        self.embed_size = embed_size
        self.head_dim = embed_size // heads

        assert (self.head_dim * heads == embed_size), "Embedding size needs to be divisible by heads"

        self.values = nn.Linear(embed_size, embed_size, bias=False)
        self.keys = nn.Linear(embed_size, embed_size, bias=False)
        self.queries = nn.Linear(embed_size, embed_size, bias=False)
        self.fc_out = nn.Linear(embed_size, embed_size)

    def forward(self, x):
        N = x.shape[0]
        length = x.shape[1]
        value_len, key_len, query_len = length, length, length

        values = self.values(x)
        keys = self.keys(x)
        queries = self.queries(x)

        values = values.view(N, value_len, self.heads, self.head_dim).permute(0, 2, 1, 3)
        keys = keys.view(N, key_len, self.heads, self.head_dim).permute(0, 2, 1, 3)
        queries = queries.view(N, query_len, self.heads, self.head_dim).permute(0, 2, 1, 3)

        energy = torch.einsum('qhd,khd->qhk', [queries, keys])
        attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=2)

        out = torch.einsum('qhk,khd->qhd', [attention, values]).reshape(N, query_len, self.heads * self.head_dim)
        return self.fc_out(out)

class Transformer(nn.Module):
    def __init__(self, embed_size, heads, num_layers):
        super(Transformer, self).__init__()
        self.layers = nn.ModuleList([
            MultiHeadAttention(embed_size, heads) for _ in range(num_layers)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

def main():
    embed_size = 256
    heads = 8
    num_layers = 6
    model = Transformer(embed_size, heads, num_layers)
    x = torch.rand(64, 10, embed_size)  # Batch size 64, sequence length 10
    out = model(x)
    print(out.shape)

if __name__ == '__main__':
    main()