# Sentiment Analysis with Deep Learning

A PyTorch-based sentiment analysis system that uses LSTM networks to classify text sentiment. The project includes training on both IMDb and SST-2 datasets, with support for custom tokenization and BERT tokenization.

## Features

- LSTM-based sentiment classification model
- Training on IMDb movie reviews dataset
- Fine-tuning capabilities on SST-2 dataset  
- Custom tokenizer implementation
- BERT tokenizer integration
- Interactive prediction interface
- Comprehensive test suite with 100+ sample sentences
- Command line interface for real-time predictions

## Setup

```bash
# Install dependencies
pip install torch transformers datasets tqdm flask

# Train base model on IMDb
python train.py

# Fine-tune on SST-2 
python fine_tunning.py

# Run tests
python Tests.py

# Interactive predictions
python Interpret.py
```

## Usage

Training
```bash
from train import train_model

# Train on IMDb dataset
model = train_model()
```

Inference
```bash
from Tests import predict

sentiment, probabilities = predict("Your text here")
```

## Model Architecture

- Embedding layer (300d)
- LSTM layers with dropout (0.3)
- Linear output layer
- Cross-entropy loss for training

## Performance

- Training accuracy: ~85-90%
- Validation accuracy: ~83-87%