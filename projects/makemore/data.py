import os
from typing import Literal
import torch
from torch.utils.data import Dataset

_current_dir = os.path.dirname(__file__)

def read_names(file_name: str|None = None):
    file_name = file_name or os.path.join(_current_dir, 'names.txt')
    with open(file_name, mode='rt', encoding='utf-8') as f:
        names = f.read().splitlines()
    return names


# names = read_names()
# chrs = ['.'] + sorted(set(''.join(names)))
# print(len(chrs))

# itos = {i:s for i,s in enumerate(chrs)}
# stoi = {s:i for i,s in enumerate(chrs)}

class NamesEncoding:
    def __init__(self, vocab):
        self.vocab = vocab
        self.vocab_size = len(vocab)
        self.itos = {i:s for i,s in enumerate(vocab)}
        self.stoi = {s:i for i,s in enumerate(vocab)}

    def encode(self, text: str|list):
        if isinstance(text, str):
            return [self.stoi[s] for s in text]
        elif isinstance(text, list):
            return[[self.stoi[s] for s in name] for name in text]

    def decode(self, ids: list[int] | list[list[int]]):
        if not isinstance(ids, list):
            raise TypeError(ids)
        
        if isinstance(ids[0], int):
            return ''.join(self.itos[i] for i in ids if isinstance(i, int))
        elif isinstance(ids[0], list):
            return [
                ''.join(self.itos[i] for i in id_) for id_ in ids if isinstance(id_, list)
            ]

# tokenizer = NamesEncoding(chrs)
# encoded = tokenizer.encode(names)
# print(tokenizer.decode(encoded))


# def build_dataset(names, context_length):
    

class NamesDataset(Dataset):
    def __init__(self, file_name: str|None = None, split: Literal['train', 'val', 'test'] = 'train', context_size: int = 4):
        names = read_names(file_name=file_name)
        vocab = ['.'] + sorted(set(''.join(names)))
        names_size = len(names)
        train_size = int(0.8*names_size)
        val_size = int(0.9*names_size)
        if split == 'train':
            names = names[:train_size]
        elif split == 'val':
            names = names[train_size:val_size]
        elif split == 'test':
            names = names[val_size:]
        else:
            raise ValueError(split)
        self.names = names

        tokenizer = NamesEncoding(vocab)
        X = []
        Y = []

        for name in self.names:
            name = '.' + name + '.'
            encoded = tokenizer.encode(name)
            if len(encoded) < context_size:
                encoded.extend([0]*(context_size - len(encoded)+1)) # type: ignore
            # print(len(encoded))
            for i in range(len(encoded)-context_size):
                X.append(encoded[i:i+context_size])
                Y.append(encoded[i+1:i+context_size+1])

        self.X = torch.tensor(X)
        self.Y = torch.tensor(Y)

        # self.X_decoded = tokenizer.decode(X)
        # self.Y_decoded = tokenizer.decode(Y)

    def __getitem__(self, index):
        return self.X[index], self.Y[index]
    
    def __len__(self):
        return self.X.shape[0]
    
