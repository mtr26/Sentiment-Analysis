from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from transformers import BertTokenizer
import torch as th

device = "cpu"
dataset = load_dataset("imdb")
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

class IMDBDataset(Dataset):
    def __init__(self, split):
        self.dataset = dataset[split]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        text = item['text']
        label = item['label']
        return text, label

def collate_batch(batch):
    texts, labels = zip(*batch)
    texts = list(texts)  # Ensure texts is a list of strings
    encodings = tokenizer(texts, truncation=True, padding=True, return_tensors="pt")
    input_ids = encodings['input_ids'].to(device)
    attention_mask = encodings['attention_mask'].to(device)
    labels = th.tensor(labels, dtype=th.float).to(device)
    return input_ids, attention_mask, labels


