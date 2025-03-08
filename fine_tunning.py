import torch as th
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer
from datasets import load_dataset
from model import Model
from tqdm import tqdm

device = "cpu"  # or check for CUDA if available

# Load SST-2 (GLUE) dataset
dataset = load_dataset("glue", "sst2")

# Use BERT tokenizer for tokenization
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

class SST2Dataset(Dataset):
    def __init__(self, split="train"):
        self.dataset = dataset[split]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        # 'sentence' is the key for the text and 'label' is the key for the label
        text = item["sentence"]
        label = item["label"]
        return text, label

def collate_batch(batch):
    texts, labels = zip(*batch)
    # Use the tokenizer to encode the batch; adjust max_length as needed
    encodings = tokenizer(
        list(texts),
        truncation=True,
        padding=True,
        return_tensors="pt"
    )
    input_ids = encodings["input_ids"].to(device)
    attention_mask = encodings["attention_mask"].to(device)
    labels = th.tensor(labels, dtype=th.long).to(device)
    return input_ids, attention_mask, labels

# Create dataset and dataloader
train_dataset = SST2Dataset(split="train")
val_dataset = SST2Dataset(split="validation")

batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_batch)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_batch)

# Set model hyperparameters
vocab_size = len(tokenizer)  # using tokenizer's vocabulary size for embedding layer
embed_size = 300
hidden_size = 300
num_classes = 2          # binary classification (negative=0, positive=1)
num_layers = 2
lr = 1e-3
epochs = 5

model = th.load("Model_Complex", map_location=device)
model = model.to(device)

optimizer = optim.Adam(model.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()

def evaluate_accuracy(model, data_loader):
    model.eval()
    total = 0
    correct = 0
    with th.no_grad():
        for input_ids, attention_mask, labels in data_loader:
            logits = model(input_ids)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    model.train()
    return correct / total

print("Starting training on SST-2")
for epoch in range(epochs):
    epoch_loss = 0.0
    for input_ids, attention_mask, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
        optimizer.zero_grad()
        logits = model(input_ids)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        
    avg_loss = epoch_loss / len(train_loader)
    val_acc = evaluate_accuracy(model, val_loader)
    print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}, Validation Accuracy: {val_acc*100:.2f}%")

# Save the trained model
th.save(model, "Model_Complex_trained")
