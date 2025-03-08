import re
import torch as th


class Tokenizer:
    def __init__(self):
        self.pad_token = '[PAD]'
        self.unk_token = '[UNK]'
        self.bos_token = '[BOS]'
        self.eos_token = '[EOS]'
        self.special_tokens = [self.pad_token, self.unk_token, self.bos_token, self.eos_token]
        self.vocab = {}
        self.inv_vocab = {}
        self.create_vocab()

    def create_vocab(self):
        with open('words.txt', 'r') as f:
            self.vocab = {token.split('\n')[0].lower(): idx for idx, token in enumerate(f.readlines())} 
        self.inv_vocab = {idx: token for token, idx in self.vocab.items()}
        for token in self.special_tokens:
            if token not in self.vocab:
                self.vocab[token] = len(self.vocab)
                self.inv_vocab[len(self.vocab)] = token
    
    def remove(self, text):
        text = text.lower()
        text = re.sub(r'<[^>]+>', '', text)
        text = re.sub(r'[^a-z0-9\s]', '', text)
        return text.split()
    
    def encode(self, text, max_length=None):
        tokens = self.remove(text)
        encoded = [self.vocab.get(token, self.vocab[self.unk_token]) for token in tokens]
        if max_length is not None:
            if len(encoded) < max_length:
                pad_token_id = self.vocab[self.pad_token]
                encoded = encoded + [pad_token_id] * (max_length - len(encoded))
            else:
                encoded = encoded[:max_length]
        return th.tensor(encoded, dtype=th.long)
    
    def decode(self, indices):
        tokens = [self.inv_vocab[idx.item()] for idx in indices]
        return " ".join(tokens)


encoder = Tokenizer()

print(encoder.encode("Hello, World!"))
print(encoder.decode(encoder.encode("Hello, World!")))