import torch
import torch.nn as nn
import torch.nn.functional as F

from data import NamesDataset
from torch.utils.data import DataLoader

ds = NamesDataset()
dl = DataLoader(ds, batch_size=32, shuffle=True)

class RNN(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self.emb = nn.Embedding(config['vocab_size'], config['n_embd'])
        self.rnn = nn.RNN(config['n_embd'], config['n_hidden'], batch_first=True, num_layers=2)
        self.lm_head = nn.Linear(config['n_hidden'], config['vocab_size'])

    def forward(self, x):
        x = self.emb(x)
        context, out = self.rnn(x)
        x = self.lm_head(context)
        return x

rnn = RNN(config={'vocab_size':27, 'context_size': 4, 'n_hidden': 50, 'n_embd': 16})
optimizer = torch.optim.SGD(rnn.parameters(), lr=0.01)

for epoch in range(10):
    tloss = 0
    for X, Y in dl:
        logits = rnn(X)
        B, T, C = logits.shape
        loss = F.cross_entropy(logits.view(B*T, C), Y.view(B*T))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        tloss += loss.item()
    print(f"{tloss/len(dl):.3f}")


class MLP(nn.Module):
    """
    takes the previous block_size tokens, encodes them with a lookup table,
    concatenates the vectors and predicts the next token with an MLP.
    """

    def __init__(self, config):
        super().__init__()
        self.block_size = config.block_size
        self.vocab_size = config.vocab_size
        self.wte = nn.Embedding(config.vocan_size+1, config.n_embd) # token embeddings table
        # +1 in the line above for a special <BLANK> token that gets inserted if encoding a token
        # before the beginning of the input sequence
        self.mlp = nn.Sequential(
            nn.Linear(self.block_size * config.n_embd, config.n_embd2),
            nn.Tanh(),
            nn.Linear(config.n_embd2, self.vocab_size)
        )

    def get_block_size(self):
        return self.block_size
    
    def forward(self, idx, targets = None):

        # gather the word embeddings of the previous 3 words
        embs = []
        for k in range(self.block_size):
            tok_emb = self.wte(idx) # token embeddings of shape (b, t, n_embd)
            idx = torch.roll(idx, 1, 1)
            idx[:, 0] = self.vocab_size # special <BLANK> token
            embs.append(tok_emb)

        # concat all of the embeddings together and pass through an MLP
        x = torch.cat(embs, -1) # (b, t, n_embd * block_size)
        logits = self.mlp(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)

        return logits, loss
    

class Bigram(nn.Module):
    """
    Bigram Language Model 'neural net', simply a lookup table of logits for the
    next character given a previous character.
    """

    def __init__(self, config):
        super().__init__()
        n = config.vocab_size
        self.logits = nn.Parameter(torch.zeros((n, n)))

    def get_block_size(self):
        return 1 # this model only needs one previous character to predict the next
    
    def forward(self, idx, targets=None):
        # forward pass
        logits = self.logits[idx]

        # if we are given some desired targets also calculate the loss
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
