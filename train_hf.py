from torch.utils.data import DataLoader, Dataset
import torch as th
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from model import Model
from datasets import load_dataset
from transformers import BertTokenizer
from encoder import Tokenizer

device = "cpu"#"mps" if th.backends.mps.is_available() else "cpu"


dataset = load_dataset("imdb")

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
custom_tokenizer = Tokenizer()



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

    def encode(self, text, max_length=None):
        tokens = self.remove(text)
        encoded = [self.vocab.get(token, self.vocab[self.unk_token]) for token in tokens]
        if max_length is not None:
            if len(encoded) < max_length:
                pad_token_id = self.vocab[self.pad_token]
                # Append the pad token until reaching max_length
                encoded += [pad_token_id] * (max_length - len(encoded))
            else:
                # Trim the encoded list to max_length
                encoded = encoded[:max_length]
        return th.tensor(encoded, dtype=th.long)
    
def test(n):
    arr = [0]*2
    arr[n] = 1
    return arr

def collate_batch(batch):
    texts, labels = zip(*batch)
    # Determine a max length from the batch (or you can set a fixed value)
    max_length = max(len(custom_tokenizer.remove(text)) for text in texts)
    encoded_texts = [custom_tokenizer.encode(text, max_length=max_length) for text in texts]
    input_ids = th.stack(encoded_texts).to(device)
    labels = th.tensor([test(i) for i in labels], dtype=th.float).to(device)
    return input_ids, labels



train_dataset = IMDBDataset(split="train")

batch_size = 128
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_batch)
vocab_size = len(custom_tokenizer.vocab)
embed_size = 100
hidden_size = 100
num_classes = 2
num_layers = 1
lr = 1e-3
epochs = 5

model = Model(vocab_size, hidden_size, num_classes, num_layers).to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)
criterion = nn.BCELoss()



def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


print(f"The model has {count_parameters(model):,} trainable parameters")

def evaluate_accuracy(model, data_loader):
    model.eval()
    total = 0
    correct = 0
    with th.no_grad():
        for input_ids, label_batch in data_loader:
            batch_size = input_ids.size(0)
            # initialize hidden state for the current batch
            logits = model(input_ids)
            # get predicted class from logits (assuming output shape [B, num_classes])
            preds = logits.argmax(dim=1)
            correct += (preds == th.tensor([0 if i[0] == 1 else 1 for i in label_batch])).sum().item()
            total += batch_size
    model.train()
    return correct / total

model.train()
for epoch in tqdm(range(epochs)):
    epoch_loss = 0.0
    for input_ids, label_batch in train_loader:
        batch_size = input_ids.size(0)
        optimizer.zero_grad()
        logits = model(input_ids)
        loss = criterion(logits, label_batch)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    
    train_acc = evaluate_accuracy(model, train_loader)
    print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss / len(train_loader):.12f}, Accuracy: {train_acc*100:.2f}%")


th.save(model, "Model")

def predict(sentence):
    model.eval()
    encodings = tokenizer(sentence, truncation=True, padding=True, return_tensors="pt")
    input_ids = encodings['input_ids'].to(device)
    attention_mask = encodings['attention_mask'].to(device)
    
    with th.no_grad():
        logits = model(input_ids)

    probabilities = th.softmax(logits, dim=1).squeeze()
    predicted_label = probabilities.argmax().item()

    label_mapping = {0: "Negative", 1: "Positive"}
    predicted_sentiment = label_mapping[predicted_label]

    return predicted_sentiment, probabilities


running = True

while running:
    input_ = input("-> ")
    if input_ == "q":
        running = False
    else:
        print(predict(input_))

