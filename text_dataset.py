import torch
from torch.utils.data import Dataset


class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, seq_length=50):
        self.seq_length = seq_length
        self.texts = []
        self.tokenizer = tokenizer

        for text in texts:
            tokens = tokenizer.encode(text, add_special_tokens=False)
            self.texts.extend(tokens)

    def __len__(self):
        return len(self.texts) - self.seq_length

    def __getitem__(self, idx):
        input_seq = self.texts[idx:idx + self.seq_length]
        target_seq = self.texts[idx + 1:idx + self.seq_length + 1]
        return torch.tensor(input_seq), torch.tensor(target_seq)
    
