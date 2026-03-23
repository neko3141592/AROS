import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(MultiHeadAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (self.head_dim * heads == embed_size), "Embedding size must be divisible by heads"

        self.values = nn.Linear(embed_size, embed_size, bias=False)
        self.keys = nn.Linear(embed_size, embed_size, bias=False)
        self.queries = nn.Linear(embed_size, embed_size, bias=False)
        self.fc_out = nn.Linear(embed_size, embed_size)

    def forward(self, x):
        N = x.shape[0]  # number of examples
        length = x.shape[1]  # length of the input sequence

        # Split the embedding into heads
        values = self.values(x)
        keys = self.keys(x)
        queries = self.queries(x)

        values = values.view(N, length, self.heads, self.head_dim)
        keys = keys.view(N, length, self.heads, self.head_dim)
        queries = queries.view(N, length, self.heads, self.head_dim)

        # Transpose to get dimensions (N, heads, length, head_dim)
        values = values.permute(0, 2, 1, 3)
        keys = keys.permute(0, 2, 1, 3)
        queries = queries.permute(0, 2, 1, 3)

        # Scaled dot-product attention
        energy = torch.einsum("nqhd,nkhd->nqhk", [queries, keys])
        attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)

        out = torch.einsum("nqhk,nvhd->nqhd", [attention, values]).reshape(N, length, self.heads * self.head_dim)

        return self.fc_out(out)

class FeedForward(nn.Module):
    def __init__(self, embed_size, ff_hidden_size):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(embed_size, ff_hidden_size)
        self.fc2 = nn.Linear(ff_hidden_size, embed_size)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))

class EncoderLayer(nn.Module):
    def __init__(self, embed_size, heads, ff_hidden_size, dropout):
        super(EncoderLayer, self).__init__()
        self.attention = MultiHeadAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.feed_forward = FeedForward(embed_size, ff_hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attention = self.attention(x)
        x = self.dropout(self.norm1(attention + x))
        forward = self.feed_forward(x)
        x = self.dropout(self.norm2(forward + x))
        return x

class Encoder(nn.Module):
    def __init__(self, embed_size, heads, ff_hidden_size, num_layers, dropout):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList([
            EncoderLayer(embed_size, heads, ff_hidden_size, dropout) for _ in range(num_layers)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

# Example usage
if __name__ == '__main__':
    embed_size = 512  # Embedding size
    heads = 8  # Number of heads
    ff_hidden_size = 2048  # Feed forward hidden size
    num_layers = 6  # Number of layers
    dropout = 0.1  # Dropout rate

    encoder = Encoder(embed_size, heads, ff_hidden_size, num_layers, dropout)
    dummy_input = torch.rand(64, 10, embed_size)  # Batch size 64, sequence length 10
    output = encoder(dummy_input)
    print(output.shape)  # Should output (64, 10, 512)
