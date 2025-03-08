import torch as th
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from model import Model
from datasets import load_dataset
from transformers import BertTokenizer

device = "cpu"#"mps" if th.backends.mps.is_available() else "cpu"

model = th.load("Model_Complex_trained", map_location=device)



model = model.to(device)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


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