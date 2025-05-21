
import torch
from torch import nn

class SentimentRNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim):
        super().__init__()
        # Embedding layer: converts vocab to dense vectors of size embed_dim
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx= 0)
        # RNN layer: processes sequences of embeddings, outputs hidden states of size hidden_dim
        self.rnn = nn.RNN(embed_dim, hidden_dim, batch_first=True)
        # Fully connected layer: maps the final hidden state to output_dim (e.g. number of classes)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # Pass input indices through embedding layer -> shape: (batch_size, seq_len, embed_dim)
        embedded = self.embedding(x)
        # Pass embeddings through RNN -> output (all hidden states), hidden (last hidden state)
        output, hidden = self.rnn(embedded)
        # Use last hidden state for classification; squeeze removes the extra dimension -> shape: (batch_size, hidden_dim)
        out = self.fc(hidden.squeeze(0))
        # Return the logits (unnormalized scores) for each class
        return out