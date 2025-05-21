# Put all functions into one class
import torch
from torch import nn
import re
from torch.utils.data import Dataset
import string
from collections import Counter


class IMDBDataset(Dataset):
    def __init__(self, df, max_len = 100, min_freq = 10, build_vocab = True, vocab = None):
        """
        df: pandas DataFrame with columns 'review' and 'sentiment'
        max_len: max sequence length for padding
        min_freq: minimum frequency to keep a word in vocab
        build_vocab: True if building vocab from df (train), False for test/new data
        """
        self.max_len = max_len
        self.min_freq = min_freq
        self.vocab = None
        # Encode sentiment to binary labels
        self.labels = df['sentiment'].values
        # Clean and tokenize labels
        self.texts = df['review'].apply(self.clean_text).apply(self.tokenize_line).tolist()

        # Build vocab if required for training
        if build_vocab:
            self.vocab = self.build_vocab(self.texts, self.min_freq)
        else:
            if vocab is None:
                raise ValueError("Vocab must be provided if build_vocab is False")
            self.vocab = vocab
        self.encoded_texts = self.encode_and_pad(self.texts, self.vocab, self.max_len)
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        # Return encoded tensor and label tensor
        return torch.tensor(self.encoded_texts[idx], dtype = torch.long), torch.tensor(self.labels[idx], dtype = torch.long)
    @staticmethod
    def clean_text(text):
        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)
        # Lower text just in case
        text = text.lower()
        # Remove punctuation
        text = text.translate(str.maketrans("", "", string.punctuation)) # str.maketrans is much faster
        # Remove digits
        text = re.sub(r'\d+', '', text)
        # Remove extra space
        text = ' '.join(text.split())
        return text
    @staticmethod
    def tokenize_line(line):
        words = line.lower().split(' ')
        return words 
    # Build vocabulary
    @staticmethod
    def build_vocab(tokenized_texts, min_freq = 2):
        counter = Counter()
        for tokens in tokenized_texts:
            counter.update(tokens)
            # print(counter)
        vocab = {"<pad>": 0, "<unk>": 1}
        for word, freq in counter.items():
            # print(f"{word}: {freq}")
            if freq >= min_freq:
                vocab[word] = len(vocab)
                # print(f"Final Vocab: {vocab}")
        return vocab
    def encode_and_pad(self, tokenized_texts, vocab, max_len):
        encoded = []
        for tokens in tokenized_texts:
            enc = [vocab.get(token, 1) for token in tokens]
            # pad or truncate
            if len(enc) < max_len:
                enc.extend([0] * (max_len - len(enc)))
            else:
                enc = enc[:max_len]
            encoded.append(enc)
        return encoded
    def encode_text(self, text):
        # Clean, tokenize, encode, and a pad a single string (for new data)
        clean = self.clean_text(text)
        tokens = self.tokenize_line(clean)
        enc = [self.vocab.get(token, 1) for token in tokens]
        if len(enc) < self.max_len:
            enc.extend([0] * (self.max_len - len(enc)))
        else:
            enc = enc[:self.max_len]
        return torch.tensor(enc, dtype = torch.long)