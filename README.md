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
pip3 install --no-cache-dir -r requirements.txt

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
python3 train.py \
    --params_file <path_to_params.json> \
    --num_of_epchs <number_of_epochs> \
    --path <output_model_path> \
    --pretrained <pretrained_model_path> \
    --dataset <dataset_name>
```

Example
```bash
# Train from scratch on IMDB dataset
python3 train.py --params_file params.json --num_of_epchs 100 --path model.pth --pretrained None --dataset imdb

# Fine-tune pretrained model on SST2 dataset
python3 train.py --params_file params.json --num_of_epchs 50 --path model_finetuned.pth --pretrained model.pth --dataset sst2
```

Testing
```bash
# Run evaluation tests
python3 Tests.py
```

Interactive Predictions
```bash
# Start interactive prediction interface
python3 Interpret.py --path model_path
```



## Model Architecture

- Embedding layer (300d)
- LSTM layers with dropout (0.3)
- Linear output layer
- Cross-entropy loss for training

## Performance

- Training accuracy: ~85-90%
- Validation accuracy: ~83-87%