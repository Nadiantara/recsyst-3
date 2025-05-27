# RecSys CLI (BERT4Rec Implementation)

A comprehensive command-line tool for recommender systems based on BERT4Rec, supporting preprocessing, training, evaluation, and ablation studies.

**Repository:** https://github.com/HanMyatThu/rmr_a2

## Overview

RecSys CLI is a versatile command-line tool for recommendation systems using the BERT4Rec architecture. It provides a complete pipeline from data preprocessing to model training, evaluation, and hyperparameter optimization through ablation studies.

## Features

- **Data Preprocessing**: Converts MovieLens 1M dataset into usable format for model training
- **BERT4Rec Model**: Implementation of the BERT4Rec architecture for sequential recommendation
- **Training Pipeline**: Complete training workflow with early stopping and learning rate scheduling
- **Evaluation**: Standard metrics (Recall@K, NDCG@K) with a reranking evaluation approach
- **Ablation Studies**: Easily run hyperparameter optimization with visualization
- **Interactive Mode**: Guided workflow through the entire pipeline

## Requirements

- Python 3.7+
- PyTorch
- pandas
- numpy
- tqdm
- matplotlib
- requests

## Installation

```bash
git clone https://github.com/HanMyatThu/rmr_a2.git
cd rmr_a2
pip install -r requirements.txt
```

## Usage

### Interactive Mode

The easiest way to use the tool is through interactive mode:

```bash
python rec_sys_cli.py
```

This will guide you through:
1. Downloading and preprocessing the MovieLens 1M dataset
2. Training a BERT4Rec model
3. Evaluating the model's performance

> **Note:** The default parameter values provide a good starting point but are not optimized. For the best hyperparameters as referenced in our report, run the ablation study first (see below).

### Direct Commands

#### 1. Preprocessing

```bash
python rec_sys_cli.py preprocess --raw_data_path ./dataset/ratings.dat --processed_data_dir ./processed_data --seq_length 50
```

#### 2. Training

```bash
python rec_sys_cli.py train --processed_data_dir ./processed_data --model_path model.pt --epochs 10 --batch_size 64 --lr 1e-3 --hidden_size 256 --num_heads 4 --num_layers 4 --dropout 0.2
```

#### 3. Evaluation

```bash
python rec_sys_cli.py evaluate --processed_data_dir ./processed_data --model_path model.pt --k 10 --neg_samples 99
```

#### 4. Ablation Study

```bash
python rec_sys_cli.py ablation --processed_data_dir ./processed_data --ablation_output_dir ./ablation_results
```

## Parameters

### Common Arguments
- `--processed_data_dir`: Directory for processed data (default: './processed_data')
- `--model_path`: Path to save/load the model (default: 'model.pt')
- `--seed`: Random seed (default: 42)
- `--seq_length`: Maximum sequence length (default: 50)

### Training Arguments
- `--epochs`: Number of training epochs (default: 10)
- `--batch_size`: Batch size for training/validation (default: 64)
- `--lr`: Learning rate (default: 1e-3)
- `--hidden_size`: Model hidden size (default: 256)
- `--num_heads`: Number of attention heads (default: 4)
- `--num_layers`: Number of transformer layers (default: 4)
- `--dropout`: Dropout rate (default: 0.2)
- `--mask_prob`: Probability for masking (default: 0.15)
- `--early_stopping_patience`: Patience for early stopping (default: 5)

### Evaluation Arguments
- `--k`: K value for Recall@K and NDCG@K (default: 10)
- `--neg_samples`: Number of negative samples for evaluation (default: 99)

## BERT4Rec Architecture

The implementation uses the Transformer architecture with:
- Item embeddings
- Positional embeddings (sinusoidal)
- Multi-head self-attention
- GELU activation
- Masked language model training objective

## Ablation Studies

The tool supports ablation studies on key hyperparameters:
- Batch size
- Dropout rate
- Hidden size
- Learning rate
- Number of transformer layers
- Number of negative samples
- Training epochs

Results are saved as CSV files and visualized with matplotlib.

### Optimal Hyperparameters

While the default values provide a functional model, they are not optimized for the best performance. We **strongly recommend** running the ablation study to identify the optimal hyperparameters for your specific use case:

```bash
python rec_sys_cli.py ablation
```

The ablation study will systematically test different hyperparameter combinations and output:
1. A CSV file with detailed results (`ablation_results.csv`)
2. Visualization plots comparing default vs. best values for each hyperparameter
3. Time vs. performance metrics scatter plots

After running the ablation study, check the `ablation_results` directory to identify the best parameters to use for your final model training. The parameters used in our research report were determined through this ablation process and significantly outperform the default values.

## Data Format

The tool expects the MovieLens 1M dataset format:
- User ID::Movie ID::Rating::Timestamp
- Automatically downloads dataset if not found

## Examples

### Basic Training and Evaluation
```bash
# Preprocess data
python rec_sys_cli.py preprocess

# Train with default parameters
python rec_sys_cli.py train

# Evaluate model
python rec_sys_cli.py evaluate
```

### Custom Configuration
```bash
# Train with custom parameters
python rec_sys_cli.py train --batch_size 128 --hidden_size 512 --num_layers 6 --lr 5e-4

# Evaluate with different K value
python rec_sys_cli.py evaluate --k 20
```

### Using Ablation-Optimized Parameters
```bash
# First run ablation study
python rec_sys_cli.py ablation

# Then train with the best parameters identified from ablation results
# Example (your optimal values may differ):
python rec_sys_cli.py train --batch_size 128 --hidden_size 256 --num_layers 4 --dropout 0.1 --lr 5e-4
```

## Acknowledgements

This implementation is based on:
- [BERT4Rec: Sequential Recommendation with Bidirectional Encoder Representations from Transformer](https://github.com/FeiSun/BERT4Rec)
- [BERT4Rec-VAE-Pytorch](https://github.com/jaywonchung/BERT4Rec-VAE-Pytorch)
- [Recommender Transformer](https://github.com/CVxTz/recommender_transformer)

