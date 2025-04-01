from glob import glob
import json

import torch
from torch.utils.data import TensorDataset


def get_all_stories(data_dir):
    end_of_file = '<|endoffile|>'
    file_names = glob('story/*.txt', root_dir=data_dir)

    train_stories = end_of_file
    val_stories = end_of_file
    test_stories = end_of_file

    for file_name in file_names[:100]:
        with open(data_dir / file_name, 'rt', encoding='utf-8') as f:
            train_stories += f.read()
        train_stories += end_of_file
    

    for file_name in file_names[100:128]:
        with open(data_dir / file_name, 'rt', encoding='utf-8') as f:
            val_stories += f.read()
        val_stories += end_of_file


    for file_name in file_names[128:]:
        with open(data_dir / file_name, 'rt', encoding='utf-8') as f:
            test_stories += f.read()
        test_stories += end_of_file

    return train_stories, val_stories, test_stories


def get_dataset(text, tokenizer, block_size=3):
    encoded = tokenizer.encode(text)
     
    X = []
    Y = []

    for i in range(len(encoded)-block_size):
        context = encoded[i:i+block_size]
        target = encoded[i+1:i+block_size+1]
        X.append(context)
        Y.append(target)

    X = torch.tensor(X)
    Y = torch.tensor(Y)
    return TensorDataset(X, Y)


class CharacterLevelTokenizer:
    def __init__(self, vocab_file:str = 'chars.json', special_token:str="<|endoffile|>", special_token_id:int=0):
        self.special_token = special_token
        self.special_token_id = special_token_id

        with open(vocab_file, 'rt', encoding='utf-8') as f:
            vocab = json.load(f)

        self.itos = {i:s for i, s in enumerate(vocab, 1)}
        self.stoi = {s:i for i, s in self.itos.items()}
        self.vocab_size = len(vocab) + 1 # add 1 for special token

    def encode(self, text):
        tokens = []
        
        i = 0
        while i < len(text):
            if text[i:].startswith(self.special_token):
                tokens.append(self.special_token_id)
                i += len(self.special_token)
            else:
                tokens.append(self.stoi[text[i]])
                i += 1
        
        return tokens

    def decode(self, tokens):
        text = "".join(self.special_token if token == self.special_token_id else self.itos[token] for token in tokens)
        
        return text
    
