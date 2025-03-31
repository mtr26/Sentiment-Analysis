import torch as th
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from model import Model
from datasets import load_dataset
from transformers import BertTokenizer


GREEN = '\033[92m'
RED = '\033[91m'
BLUE = '\033[94m'
RESET = '\033[0m'

device = "cpu"

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


def print_welcome():
    print(f"{BLUE}====================================")
    print("   Sentiment Analysis Interpreter")
    print("====================================")
    print("\nEnter a sentence to analyze its sentiment")
    print("Type 'q' or 'quit' to exit")
    print(f"====================================={RESET}\n")


import argparse
def main():
    print_welcome()
    running = True
    parser = argparse.ArgumentParser()
    parser.add_argument('--path')
    args = parser.parse_args()
    model = th.load(args.path, map_location=device, weights_only=False)
    model = model.to(device)
    while running:
        try:
            user_input = input(f"{BLUE}Enter text → {RESET}").strip()
            
            if user_input.lower() in ['q', 'quit']:
                print(f"\n{BLUE}Goodbye!{RESET}")
                running = False
                continue
                
            if not user_input:
                print(f"{RED}Please enter some text to analyze{RESET}")
                continue

            sentiment, confidence = predict(user_input, model)
            color = GREEN if sentiment == "Positive" else RED
            print(f"\nSentiment: {color}{sentiment}{RESET}")
            print(f"Confidence: {color}{confidence:.1f}%{RESET}\n")

        except KeyboardInterrupt:
            print(f"\n{BLUE}Goodbye!{RESET}")
            running = False
        except Exception as e:
            print(f"{RED}Error: {str(e)}{RESET}")


def predict(sentence, model):
    model.eval()
    encodings = tokenizer(sentence, truncation=True, padding=True, return_tensors="pt")
    input_ids = encodings['input_ids'].to(device)
    attention_mask = encodings['attention_mask'].to(device)
    
    with th.no_grad():
        logits = model(input_ids)

    probabilities = th.softmax(logits, dim=1).squeeze()
    predicted_label = probabilities.argmax().item()
    confidence = probabilities[predicted_label].item() * 100

    label_mapping = {0: "Negative", 1: "Positive"}
    predicted_sentiment = label_mapping[predicted_label]

    return predicted_sentiment, confidence


if __name__ == "__main__":
    main()