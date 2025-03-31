
from torch.utils.data import DataLoader, Dataset
import torch as th
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from model import Model
from datasets import load_dataset
from transformers import BertTokenizer

device = "cpu"#"mps" if th.backends.mps.is_available() else "cpu"


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




def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)




def evaluate_accuracy(model, data_loader):
    model.eval()
    total = 0
    correct = 0
    with th.no_grad():
        for input_ids, attention_mask, label_batch in data_loader:
            batch_size = input_ids.size(0)
            logits = model(input_ids)
            preds = logits.argmax(dim=1)
            correct += (preds == label_batch).sum().item()
            total += batch_size
    model.train()
    return correct / total



def train(num_epochs : int, model_path : str, model : Model, test_loader : DataLoader):
    print(f"The model has {count_parameters(model):,} trainable parameters")
    model.train()
    for epoch in tqdm(range(num_epochs)):
        epoch_loss = 0.0
        for input_ids, attention_mask, label_batch in train_loader:
            batch_size = input_ids.size(0)
            optimizer.zero_grad()
            logits = model(input_ids)
            loss = criterion(logits, label_batch.long())
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        train_acc = evaluate_accuracy(model, test_loader)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss / len(test_loader):.12f}, Accuracy: {train_acc*100:.2f}%")
    th.save(model, model_path)



import argparse
import json

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--params_file')
    parser.add_argument('--num_of_epchs')
    parser.add_argument('--path')
    parser.add_argument('--pretrained')
    parser.add_argument('--dataset')

    args = parser.parse_args()

    with open(args.params_file, 'r+') as f:
        data = json.load(f)
    
    try:
        batch_size = data["batch_size"]
        embed_size = data["embed_size"]
        hidden_size = data["hidden_size"]
        num_classes = data["num_classes"]
        num_layers = data["num_layers"]
        lr = data["learning_rate"]
    except:
        raise Exception("Failed to read the json file. Please check the format.")
    
    if args.dataset == 'imdb':
        from IMDB_Loader import *
        train_dataset = IMDBDataset(split="train")
        test_dataset = IMDBDataset(split="test")
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_batch)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_batch)
    elif args.dataset == 'sst2':
        from SST2_Loader import *
        train_dataset = SST2Dataset(split="train")
        test_dataset = SST2Dataset(split="validation")
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_batch)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_batch)
    else:
        raise Exception(f"Invalid dataset '{args.dataset}'. Please choose between 'imdb' and 'sst2'.")

    
    if args.pretrained == 'None':
        vocab_size = len(tokenizer)
        model = Model(vocab_size, hidden_size, num_classes, num_layers)
    else:
        model = th.load(args.pretrained, map_location=device)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    train(int(args.num_of_epchs), args.path, model, test_loader)



    
    