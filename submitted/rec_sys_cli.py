import argparse
import pandas as pd
import numpy as np
import json
import os
import math # For BERT4Rec sinusoidal encoding & evaluate_rerank
import random # For set_seed
import time # For ablation study timing
import requests # For downloading dataset
import zipfile # For extracting dataset
from io import BytesIO # For handling zip in memory

import torch
import torch.nn as nn # For BERT4Rec
import torch.nn.functional as F # For training loss
from torch.utils.data import Dataset, DataLoader # For datasets and dataloaders
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR
from torch.nn import TransformerEncoder, TransformerEncoderLayer # For BERT4Rec

from tqdm import tqdm # For training progress
import matplotlib.pyplot as plt # For ablation plotting

# Constants from preprocess.py
PAD_TOKEN = 0
SEQ_LENGTH = 50
DEFAULT_DATA_PATH = './dataset/ratings.dat'
DEFAULT_OUTPUT_DIR = './processed_data'

# --- Dataset Download Function ---
MOVIELENS_URL = "http://files.grouplens.org/datasets/movielens/ml-1m.zip"

def download_movielens_1m(target_raw_data_dir: str, ratings_filename: str = "ratings.dat"):
    """Downloads and extracts the MovieLens 1M ratings.dat file."""
    os.makedirs(target_raw_data_dir, exist_ok=True)
    target_file_path = os.path.join(target_raw_data_dir, ratings_filename)

    if os.path.exists(target_file_path):
        print(f"Dataset {ratings_filename} already exists at {target_file_path}.")
        return True

    print(f"Downloading MovieLens 1M dataset from {MOVIELENS_URL}...")
    try:
        response = requests.get(MOVIELENS_URL, stream=True)
        response.raise_for_status() # Raise an exception for HTTP errors
        
        print("Extracting ratings.dat...")
        with zipfile.ZipFile(BytesIO(response.content)) as z:
            # The file inside the zip is typically 'ml-1m/ratings.dat'
            # We want to extract it directly to target_raw_data_dir/ratings_filename
            member_name = 'ml-1m/ratings.dat'
            if member_name in z.namelist():
                with z.open(member_name) as source, open(target_file_path, 'wb') as target:
                    target.write(source.read())
                print(f"Successfully downloaded and extracted {ratings_filename} to {target_file_path}")
                return True
            else:
                print(f"Error: Could not find {member_name} in the downloaded zip file.")
                return False
    except requests.exceptions.RequestException as e:
        print(f"Error downloading dataset: {e}")
        return False
    except zipfile.BadZipFile:
        print("Error: Downloaded file is not a valid zip file or is corrupted.")
        return False
    except Exception as e:
        print(f"An unexpected error occurred during download/extraction: {e}")
        return False

def ensure_dataset_exists(raw_data_file_path: str, raw_data_dir: str):
    """Checks if the raw dataset exists, and if not, prompts the user to download it."""
    if not os.path.exists(raw_data_file_path):
        print(f"Raw data file not found at {raw_data_file_path}.")
        if input("Do you want to download the MovieLens 1M dataset? (yes/no): ").strip().lower() == 'yes':
            if not download_movielens_1m(target_raw_data_dir=raw_data_dir):
                print("Failed to obtain the dataset. Please ensure it's manually placed or try again.")
                return False # Indicate failure
        else:
            print("Dataset not available. Cannot proceed with preprocessing.")
            return False # Indicate failure
    return True # Dataset exists or was successfully downloaded

# Helper function from preprocess.py
def padOrTruncate(seq, max_len=SEQ_LENGTH):
    if len(seq) >= max_len:
        return seq[-max_len:]
    return [PAD_TOKEN] * (max_len - len(seq)) + seq

# Preprocessing function from preprocess.py
def run_preprocess(data_path: str, output_dir: str, seq_length_setting: int):
    # Ensure dataset exists before proceeding
    raw_data_dir = os.path.dirname(data_path)
    if not ensure_dataset_exists(data_path, raw_data_dir):
        return # Stop preprocessing if dataset is not available

    os.makedirs(output_dir, exist_ok=True)

    # load and filter positive interactions (rating >= 4)
    df = pd.read_csv(
        data_path,
        sep=r'::',
        engine='python',
        names=['userId', 'movieId', 'rating', 'timestamp']
    )
    df = df[df.rating >= 4].copy()

    # generate chronological interaction sequences per user.
    df.sort_values(['userId', 'timestamp'], inplace=True)
    userSequence = df.groupby('userId')['movieId'].apply(list).to_dict()

    # filter out users with fewer than 5 interactions
    userSequence = {u: seq for u, seq in userSequence.items() if len(seq) >= 5}

    # map movie IDs (0 reserved for PAD)
    all_items = sorted({item for seq in userSequence.values() for item in seq})
    item2idx = {item: idx + 1 for idx, item in enumerate(all_items)}
    num_items = len(item2idx)

    train_inputs, val_inputs, test_inputs = [], [], []
    val_labels, test_labels = [], []
    train_users, val_users, test_users = [], [], []

    # split each user's sequence into train/val/test and prepare contexts & labels
    for user, seq in userSequence.items():
        # remap items
        seq = [item2idx[i] for i in seq]
        n = len(seq)
        i1 = int(n * 0.7)
        i2 = int(n * 0.85)

        # Training: use first i1 items as input
        train_seq = seq[:i1]
        train_input = padOrTruncate(train_seq)

        # Validation: input is prefix up to i1, labels are seq[i1:i2]
        val_input = padOrTruncate(seq[:i1])
        val_label = seq[i1:i2]

        # Test: input is prefix up to i2, labels are seq[i2:]
        test_input = padOrTruncate(seq[:i2])
        test_label = seq[i2:]

        train_inputs.append(train_input)
        val_inputs.append(val_input)
        test_inputs.append(test_input)
        val_labels.append(val_label)
        test_labels.append(test_label)
        train_users.append(user)
        val_users.append(user)
        test_users.append(user)

    # save inputs and labels
    np.save(os.path.join(output_dir, 'train_inputs.npy'), np.array(train_inputs, dtype=np.int32))
    np.save(os.path.join(output_dir, 'val_inputs.npy'),   np.array(val_inputs,   dtype=np.int32))
    np.save(os.path.join(output_dir, 'test_inputs.npy'),  np.array(test_inputs,  dtype=np.int32))
    np.save(os.path.join(output_dir, 'val_labels.npy'),   np.array(val_labels,   dtype=object))
    np.save(os.path.join(output_dir, 'test_labels.npy'),  np.array(test_labels,  dtype=object))
    np.save(os.path.join(output_dir, 'train_users.npy'),  np.array(train_users))
    np.save(os.path.join(output_dir, 'val_users.npy'),    np.array(val_users))
    np.save(os.path.join(output_dir, 'test_users.npy'),   np.array(test_users))

    # save metadata
    meta = {
        'item2idx':   item2idx,
        'num_items':  num_items,
        'pad_token':  PAD_TOKEN,
        'seq_length': seq_length_setting
    }
    with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:
        json.dump(meta, f)

    print(f"Preprocessing complete. Processed data saved to {output_dir}")

# --- BERT4Rec Model (from model.py) ---
class BERT4Rec(nn.Module):
    def __init__(
        self,
        num_items: int,
        hidden_size: int,
        num_heads: int,     
        num_layers: int,
        max_seq_len: int, # Changed from SEQ_LENGTH to be more dynamic
        dropout: float
    ):
        super().__init__()
        self.pad_token = PAD_TOKEN # Use global PAD_TOKEN
        self.mask_token = num_items + 1
        self.vocab_size = num_items + 2 
        self.hidden_size = hidden_size

        self.item_embeddings = nn.Embedding(
            self.vocab_size, 
            hidden_size, 
            padding_idx=self.pad_token
        )
        
        self.register_buffer(
            'position_embeddings',
            self._get_sinusoidal_encoding(max_seq_len, hidden_size)
        )

        encoder_layer = TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.layer_norm = nn.LayerNorm(hidden_size)
        self.output_layer = nn.Linear(hidden_size, self.vocab_size)

    def _get_sinusoidal_encoding(self, max_len, d_model):
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe

    def forward(self, input_ids):
        batch_size, seq_len = input_ids.size()

        item_emb = self.item_embeddings(input_ids)
        # Use actual seq_len from input_ids for position_embeddings slicing
        pos_emb = self.position_embeddings[:input_ids.size(1)].unsqueeze(0).expand(batch_size, -1, -1)
        x = item_emb + pos_emb

        pad_mask = input_ids.eq(self.pad_token)
        x = self.transformer(x, src_key_padding_mask=pad_mask)

        x = self.layer_norm(x)
        return self.output_layer(x)

# --- Dataloader components (from dataloader.py) ---
class TrainDataset(Dataset):
    def __init__(self, sequences: np.ndarray, mask_token: int, seq_length: int, mask_prob: float = 0.15):
        self.sequences = sequences
        self.mask_token = mask_token
        self.mask_prob = mask_prob
        self.seq_length = seq_length # sequences.shape[1]

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx].copy()
        labels = np.full(self.seq_length, fill_value=-100, dtype=np.int64)
        non_pad_positions = np.where(seq != PAD_TOKEN)[0] # Use global PAD_TOKEN
        if len(non_pad_positions) == 0: # Handle empty sequences if they exist
            return torch.LongTensor(seq), torch.LongTensor(labels)
            
        num_to_mask = max(1, int(len(non_pad_positions) * self.mask_prob))
        mask_positions = np.random.choice(non_pad_positions, num_to_mask, replace=False)

        for pos in mask_positions:
            labels[pos] = seq[pos]
            rand = np.random.rand()
            if rand < 0.8:
                seq[pos] = self.mask_token
            elif rand < 0.9:
                seq[pos] = np.random.randint(1, self.mask_token) # Assumes item IDs are [1, mask_token-1]
            # Else: 10% keep original
        return torch.LongTensor(seq), torch.LongTensor(labels)

class EvalDataset(Dataset):
    def __init__(self, sequences: np.ndarray, labels: np.ndarray): # seq_length removed, infer from data
        self.sequences = sequences
        self.labels = labels
        # self.seq_length = sequences.shape[1] # Not strictly needed if collate handles padding

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx].copy()
        future_items = list(self.labels[idx]) if self.labels is not None and idx < len(self.labels) else []
        return torch.LongTensor(seq), future_items

def eval_collate_fn(batch):
    seqs = torch.stack([item[0] for item in batch], dim=0)
    future_items = [item[1] for item in batch]
    return seqs, future_items
    
# --- Evaluation Function (from evaluate.py, slightly modified) ---
def evaluate_rerank(model, dataloader, device, num_items, k=10, neg_samples=99):
    model.eval()
    items_range = np.arange(1, num_items + 1) # Renamed from 'items' to avoid conflict
    recalls, ndcgs = [], []
    
    # Ensure model has pad_token and mask_token attributes
    pad_token_val = model.pad_token if hasattr(model, 'pad_token') else PAD_TOKEN
    mask_token_val = model.mask_token if hasattr(model, 'mask_token') else num_items + 1


    with torch.no_grad():
        for seqs, future_items_list in dataloader:
            seqs = seqs.to(device)
            batch_size, seq_len = seqs.size()

            masked_seqs = seqs.clone()
            mask_positions = []
            for i in range(batch_size):
                non_pad = torch.nonzero(seqs[i] != pad_token_val, as_tuple=True)[0]
                pos = non_pad[-1].item() if len(non_pad) > 0 else seq_len - 1 # Mask last if all pads or empty
                masked_seqs[i, pos] = mask_token_val
                mask_positions.append(pos)
            
            logits = model(masked_seqs)
            scores = logits[torch.arange(batch_size), mask_positions, :].cpu().numpy()

            for i, future_items in enumerate(future_items_list):
                if not future_items: # Check if list is empty
                    continue

                true_item = future_items[0]
                user_history = set(seqs[i][seqs[i] != pad_token_val].cpu().numpy().tolist()) # Exclude pad
                
                # Filter neg_pool more carefully
                neg_pool_candidates = [x for x in items_range if x != true_item and x not in user_history and x != pad_token_val and x != mask_token_val]

                if not neg_pool_candidates: # If no valid negative samples can be drawn
                    # This might happen if user history + true_item cover almost all items
                    # Or if num_items is very small.
                    # Depending on desired behavior, we could skip, or use a relaxed filtering.
                    # For now, append 0 for metrics if we can't form candidates.
                    recalls.append(0.0)
                    ndcgs.append(0.0)
                    continue

                actual_neg_samples = min(neg_samples, len(neg_pool_candidates))
                if actual_neg_samples == 0 and neg_samples > 0 : # Still no negs after trying to adjust
                     recalls.append(0.0)
                     ndcgs.append(0.0)
                     continue

                negs = np.random.choice(neg_pool_candidates, actual_neg_samples, replace=False) if actual_neg_samples > 0 else np.array([])


                candidates = np.concatenate([[true_item], negs]) if negs.size > 0 else np.array([true_item])
                
                # Ensure candidate indices are within bounds of scores
                valid_candidates = candidates[candidates < scores.shape[1]]
                if not valid_candidates.any(): # if no valid candidates remain
                    recalls.append(0.0)
                    ndcgs.append(0.0)
                    continue

                candidate_scores = scores[i, valid_candidates]
                
                rank_indices = np.argsort(-candidate_scores)
                top_k_candidates = valid_candidates[rank_indices][:k]

                recall = 1.0 if true_item in top_k_candidates else 0.0
                recalls.append(recall)

                if true_item in top_k_candidates:
                    rank = np.where(top_k_candidates == true_item)[0][0]
                    ndcg = 1.0 / math.log2(rank + 2)
                else:
                    ndcg = 0.0
                ndcgs.append(ndcg)
    
    mean_recall = np.mean(recalls) if recalls else 0.0
    mean_ndcg = np.mean(ndcgs) if ndcgs else 0.0
    return {'recall': mean_recall, 'ndcg': mean_ndcg}

# --- Training components (adapted from train.py) ---
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class EarlyStopping:
    def __init__(self, patience=5, metric_name='ndcg'):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.best_epoch = None
        self.metric_name = metric_name # e.g. 'ndcg' or 'recall'
        self.best_model_path = 'early_stop_best_model.pt' # Temp path for best model

    def __call__(self, current_metric_val, epoch, model, model_save_path):
        if self.best_score is None or current_metric_val > self.best_score:
            self.best_score = current_metric_val
            self.best_epoch = epoch
            self.counter = 0
            torch.save(model.state_dict(), self.best_model_path) # Save to temp path
            # print(f"EarlyStopping: New best score ({self.metric_name}={current_metric_val:.4f}) at epoch {epoch}. Model saved.")
        else:
            self.counter += 1
            # print(f"EarlyStopping: Score did not improve for {self.counter} epochs.")
            if self.counter >= self.patience:
                print(f"EarlyStopping: Stopping. Best score ({self.metric_name}={self.best_score:.4f}) achieved at epoch {self.best_epoch}.")
                # copy the best model to the final destination before returning True
                if os.path.exists(self.best_model_path):
                    os.replace(self.best_model_path, model_save_path)
                    print(f"Best model from epoch {self.best_epoch} saved to {model_save_path}")
                return True
        return False

# --- Ablation Study Components (adapted from ablation.py) ---
ABLATION_DEFAULTS = {
    'batch_size': 64,
    'dropout': 0.2,
    'hidden_size': 128,
    'learning_rate': 5e-4,
    'num_layers': 2,
    'neg_samples': 99, 
    'epochs': 5,
    'num_heads': 4, 
    'mask_prob': 0.15
}

ABLATION_VALUES = {
    'batch_size': [32, 64, 128],
    'dropout': [0.1, 0.2, 0.3],
    'hidden_size': [64, 128, 256],
    'learning_rate': [1e-4, 5e-4, 1e-3],
    'num_layers': [1, 2, 4], # Changed 3 to 4 to match corrected layer num
    'neg_samples': [50, 99, 150],
    'epochs': [3, 5, 8]
}

def run_ablation_train_eval(hparams, processed_data_dir, base_seed):
    # Each run within ablation should be deterministic but vary if needed by modifying seed based on hparams
    # For now, use a fixed seed for comparable runs unless specific hparam seeding is desired.
    set_seed(base_seed) 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    run_start_time = time.time()

    meta_path = os.path.join(processed_data_dir, 'metadata.json')
    if not os.path.exists(meta_path):
        print(f"Ablation Error: Metadata not found at {meta_path}. Run preprocessing.")
        return 0, 0, 0 # recall, ndcg, time
    with open(meta_path, 'r') as f:
        meta = json.load(f)
    
    num_items = meta['num_items']
    # Use seq_length from metadata for consistency, as it affects model and data loading
    model_seq_len = meta.get('seq_length', ABLATION_DEFAULTS.get('max_seq_len', 50)) 
    mask_token_id = num_items + 1

    train_inputs = np.load(os.path.join(processed_data_dir, 'train_inputs.npy'))
    val_inputs = np.load(os.path.join(processed_data_dir, 'val_inputs.npy'))
    val_labels = np.load(os.path.join(processed_data_dir, 'val_labels.npy'), allow_pickle=True)

    train_loader = DataLoader(
        TrainDataset(train_inputs, mask_token=mask_token_id, seq_length=model_seq_len, mask_prob=hparams['mask_prob']),
        batch_size=hparams['batch_size'], shuffle=True)
    val_loader = DataLoader(
        EvalDataset(val_inputs, val_labels),
        batch_size=hparams['batch_size'], shuffle=False, collate_fn=eval_collate_fn)
    
    model = BERT4Rec(
        num_items=num_items,
        hidden_size=hparams['hidden_size'],
        num_heads=hparams['num_heads'],
        num_layers=hparams['num_layers'],
        max_seq_len=model_seq_len, 
        dropout=hparams['dropout']
    ).to(device)
    
    optimizer = AdamW(model.parameters(), lr=hparams['learning_rate'])
    
    # Simplified scheduler for ablation runs, focusing on total epochs
    # scheduler = CosineAnnealingLR(optimizer, T_max=hparams['epochs'] * len(train_loader), eta_min=hparams['learning_rate']*0.01)
    # A common practice is to define warmup steps based on a few initial epochs or a fixed number of steps.
    # For simplicity in ablation, let's use a fixed small number of warmup epochs if any.
    warmup_epochs = 1 # Fixed to 1 epoch of warmup for ablation runs
    total_steps_for_run = len(train_loader) * hparams['epochs']
    warmup_steps_for_run = len(train_loader) * warmup_epochs

    def lr_lambda_ablation(current_step):
        if current_step < warmup_steps_for_run:
            return float(current_step) / float(max(1, warmup_steps_for_run))
        # After warmup, could go to cosine, or just stay constant, or a step decay.
        # For ablation, often a simpler schedule or just focusing on convergence by end of epochs is fine.
        # Here, let's just maintain LR after warmup for simplicity, or use cosine.
        # Cosine part:
        # return 0.5 * (1 + math.cos(math.pi * (current_step - warmup_steps_for_run) / (total_steps_for_run - warmup_steps_for_run)))
        # Simpler: constant after warmup
        return 1.0 

    scheduler = LambdaLR(optimizer, lr_lambda_ablation)

    for epoch in range(1, hparams['epochs'] + 1):
        model.train()
        # epoch_loss = 0 # Not tracking loss verbosely for ablation runs to reduce output
        # progress_bar = tqdm(train_loader, desc=f"Ablation Epoch {epoch}/{hparams['epochs']}", leave=False)
        for input_ids, labels in train_loader: # progress_bar removed for cleaner ablation logs
            input_ids, labels = input_ids.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(input_ids)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            if current_step < warmup_steps_for_run: # only step scheduler during warmup if using lambda that way
                 scheduler.step()
            # epoch_loss += loss.item()
        # After warmup, if using cosine, step it per epoch or per batch. The current lambda is per batch.
        if current_step >= warmup_steps_for_run and isinstance(scheduler, CosineAnnealingLR): # Example if mixing
             scheduler.step() # CosineAnnealingLR is typically stepped per epoch after warmup if warmup is separate
        elif not isinstance(scheduler, CosineAnnealingLR): # For LambdaLR that depends on global step like above
            pass # it's stepped per batch

    # No early stopping in ablation runs; train for the full specified epochs for that hparam set.
    metrics = evaluate_rerank(
        model, val_loader, device, num_items=num_items, 
        k=DEFAULT_EVAL_K, # Use default K for ablation eval consistency
        neg_samples=hparams['neg_samples']
    )
    
    total_run_time = time.time() - run_start_time
    return metrics['recall'], metrics['ndcg'], total_run_time

def plot_ablation_results(df, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    # Corrected access to ABLATION_VALUES and ABLATION_DEFAULTS
    for hp_name_plot in ABLATION_VALUES.keys(): # Changed from ABLATION_VALUES.keys()
        sub_df = df[df['hyperparameter'] == hp_name_plot]
        if sub_df.empty:
            print(f"No data for hyperparameter: {hp_name_plot}. Skipping plot.")
            continue
        
        best_row = sub_df.loc[sub_df['ndcg'].idxmax()]
        default_val_for_hp = ABLATION_DEFAULTS[hp_name_plot]
        default_perf_rows = sub_df[sub_df['value'] == default_val_for_hp]
        
        default_ndcg = default_perf_rows.iloc[0]['ndcg'] if not default_perf_rows.empty else 0
        default_value_to_display = default_perf_rows.iloc[0]['value'] if not default_perf_rows.empty else default_val_for_hp

        plt.figure(figsize=(7, 4))
        plt.bar([f'Default ({default_value_to_display})', f'Best ({best_row["value"]})'], [default_ndcg, best_row['ndcg']], color=['skyblue', 'coral'])
        plt.title(f'NDCG@10: {hp_name_plot}')
        plt.ylabel('NDCG@10')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{hp_name_plot}_default_vs_best.png'))
        plt.close()
        
    # Time vs Metric Scatter Plot
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))
    plotted_something = False
    unique_hps = df['hyperparameter'].unique()
    for hp_name_scatter in unique_hps:
        sub_df = df[df['hyperparameter'] == hp_name_scatter]
        if sub_df.empty: continue
        axs[0].scatter(sub_df['run_time'], sub_df['recall'], label=hp_name_scatter, alpha=0.7)
        axs[1].scatter(sub_df['run_time'], sub_df['ndcg'], label=hp_name_scatter, alpha=0.7)
        plotted_something = True
        
    if plotted_something:
        axs[0].set_xlabel('Run Time (s)'); axs[0].set_ylabel(f'Recall@{DEFAULT_EVAL_K}'); axs[0].set_title(f'Run Time vs Recall@{DEFAULT_EVAL_K}')
        axs[1].set_xlabel('Run Time (s)'); axs[1].set_ylabel(f'NDCG@{DEFAULT_EVAL_K}'); axs[1].set_title(f'Run Time vs NDCG@{DEFAULT_EVAL_K}')
        axs[0].legend(); axs[1].legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'time_vs_metrics_scatter.png'))
    else:
        print("No data to plot for time vs metric scatter plot.")
    plt.close()

def run_ablation_study(args):
    print("Starting ablation study...")
    results = []
    output_csv_path = os.path.join(args.ablation_output_dir, 'ablation_results.csv')
    os.makedirs(args.ablation_output_dir, exist_ok=True)

    # Use a consistent base seed for all ablation runs, or vary it if desired
    base_seed = args.seed 

    for hp_name, hp_values_list in ABLATION_VALUES.items():
        print(f"\nAblating {hp_name}...")
        current_default_hparams = ABLATION_DEFAULTS.copy() # Start with ablation defaults
        # Update with any relevant CLI args if they should override ablation defaults (e.g. data paths)
        # However, for hyperparameter ablation, ABLATION_DEFAULTS should be the baseline.

        for val in hp_values_list:
            hparams_for_run = current_default_hparams.copy()
            hparams_for_run[hp_name] = val # Set the specific hyperparameter to ablate
            
            # Fill any missing essential params from ABLATION_DEFAULTS (should mostly be covered)
            for key in ABLATION_DEFAULTS:
                if key not in hparams_for_run:
                    hparams_for_run[key] = ABLATION_DEFAULTS[key]
            
            print(f"  Running with {hp_name}={val} (params: {hparams_for_run})")
            recall, ndcg, run_time = run_ablation_train_eval(hparams_for_run, args.processed_data_dir, base_seed)
            print(f"    {hp_name}={val}: Recall@{DEFAULT_EVAL_K}={recall:.4f}, NDCG@{DEFAULT_EVAL_K}={ndcg:.4f}, Time={run_time:.1f}s")
            results.append({
                'hyperparameter': hp_name, 'value': val,
                'recall': recall, 'ndcg': ndcg, 'run_time': run_time
            })
            
    df_results = pd.DataFrame(results)
    df_results.to_csv(output_csv_path, index=False)
    print(f"\nAblation study complete. Results saved to {output_csv_path}")
    
    print("Plotting ablation results...")
    plot_ablation_results(df_results, args.ablation_output_dir)
    print(f"Plots saved in {args.ablation_output_dir}")


# --- Main entry point and CLI argument parsing ---
def main():
    parser = argparse.ArgumentParser(description="RecSys CLI - Preprocess, Train, Evaluate, or Run Ablation Study.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("action", choices=["preprocess", "train", "evaluate", "ablation", "interactive"], 
                        nargs='?', default="interactive",
                        help="The action to perform. 'interactive' will guide through steps.")
    
    # Common arguments
    parser.add_argument("--processed_data_dir", type=str, default=DEFAULT_OUTPUT_DIR, help="Directory for processed data.")
    parser.add_argument("--model_path", type=str, default='model.pt', help="Path to save/load the model.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--seq_length", type=int, default=SEQ_LENGTH, help="Max sequence length for padding and model (used in preprocessing and as fallback for model if not in metadata).")

    # Preprocessing arguments
    preproc_group = parser.add_argument_group('Preprocessing Options')
    preproc_group.add_argument("--raw_data_path", type=str, default=DEFAULT_DATA_PATH, help="Path to the raw ratings.dat file.")

    # Training arguments
    train_group = parser.add_argument_group('Training Options')
    train_group.add_argument("--epochs", type=int, default=10, help="Number of training epochs.")
    train_group.add_argument("--batch_size", type=int, default=64, help="Batch size for training and validation.")
    train_group.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    train_group.add_argument("--hidden_size", type=int, default=256, help="Model hidden size.")
    train_group.add_argument("--num_heads", type=int, default=4, help="Model number of attention heads.")
    train_group.add_argument("--num_layers", type=int, default=4, help="Model number of transformer layers.")
    train_group.add_argument("--dropout", type=float, default=0.2, help="Model dropout rate.")
    train_group.add_argument("--mask_prob", type=float, default=0.15, help="Masking probability for TrainDataset.")
    train_group.add_argument("--early_stopping_patience", type=int, default=5, help="Patience for early stopping based on validation NDCG.")

    # Evaluation arguments
    eval_group = parser.add_argument_group('Evaluation Options')
    eval_group.add_argument("--k", type=int, default=10, help="Value of K for Recall@K and NDCG@K.")
    eval_group.add_argument("--neg_samples", type=int, default=99, help="Number of negative samples for evaluation reranking.")

    # Ablation arguments
    ablation_group = parser.add_argument_group('Ablation Study Options')
    ablation_group.add_argument("--ablation_output_dir", type=str, default='ablation_results', help="Directory to save ablation study results (CSV and plots).")

    args = parser.parse_args()
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    if args.action == "interactive":
        print("Running in interactive mode...")
        
        # Check/Download dataset first for interactive mode
        raw_data_dir_interactive = os.path.dirname(args.raw_data_path)
        if not ensure_dataset_exists(args.raw_data_path, raw_data_dir_interactive):
            print("Cannot proceed without the dataset. Exiting interactive mode.")
            return
            
        # 1. Preprocessing
        print("\nStep 1: Preprocessing data...")
        run_preprocess(data_path=args.raw_data_path, output_dir=args.processed_data_dir, seq_length_setting=args.seq_length) # pass seq_length
        print("Preprocessing finished.")

        # 2. Training
        if input("\nDo you want to proceed with training the model? (yes/no): ").strip().lower() == 'yes':
            print("\nStep 2: Training model...")
            # Simplified train call for interactive mode, using args for hyperparameters
            # Ensure metadata is loaded correctly
            meta_path = os.path.join(args.processed_data_dir, 'metadata.json')
            if not os.path.exists(meta_path):
                print(f"Metadata file not found at {meta_path}. Cannot proceed with training.")
                return
            with open(meta_path, 'r') as f: meta = json.load(f)
            loaded_seq_length = meta.get('seq_length', args.seq_length)
            model_seq_len = loaded_seq_length

            train_inputs = np.load(os.path.join(args.processed_data_dir, 'train_inputs.npy'))
            val_inputs = np.load(os.path.join(args.processed_data_dir, 'val_inputs.npy'))
            val_labels = np.load(os.path.join(args.processed_data_dir, 'val_labels.npy'), allow_pickle=True)
            mask_token_id = meta['num_items'] + 1

            train_loader = DataLoader(TrainDataset(train_inputs, mask_token_id, model_seq_len, args.mask_prob), batch_size=args.batch_size, shuffle=True)
            val_loader = DataLoader(EvalDataset(val_inputs, val_labels), batch_size=args.batch_size, shuffle=False, collate_fn=eval_collate_fn)
            model = BERT4Rec(meta['num_items'], args.hidden_size, args.num_heads, args.num_layers, model_seq_len, args.dropout).to(device)
            optimizer = AdamW(model.parameters(), lr=args.lr)
            num_training_steps = args.epochs * len(train_loader)
            scheduler = CosineAnnealingLR(optimizer, T_max=num_training_steps, eta_min=args.lr*0.01)
            early_stopper = EarlyStopping(patience=args.early_stopping_patience)

            print(f"Starting training for {args.epochs} epochs (using parameters from CLI args or defaults)...")
            for epoch in range(1, args.epochs + 1):
                model.train()
                epoch_loss = 0.0
                progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", leave=False)
                for input_ids, labels_batch in progress_bar: # Renamed labels to labels_batch
                    input_ids, labels_batch = input_ids.to(device), labels_batch.to(device)
                    optimizer.zero_grad()
                    logits = model(input_ids)
                    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels_batch.view(-1), ignore_index=-100)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    scheduler.step()
                    epoch_loss += loss.item()
                    progress_bar.set_postfix({'loss': loss.item()})
                avg_epoch_loss = epoch_loss / len(train_loader)
                val_metrics = evaluate_rerank(model, val_loader, device, meta['num_items'], args.k, args.neg_samples)
                print(f"Epoch {epoch}: Avg Loss={avg_epoch_loss:.4f}, Val Recall@{args.k}={val_metrics['recall']:.4f}, Val NDCG@{args.k}={val_metrics['ndcg']:.4f}, LR={optimizer.param_groups[0]['lr']:.6f}")
                if early_stopper(val_metrics['ndcg'], epoch, model, args.model_path):
                    print("Early stopping triggered."); break
            if not early_stopper.counter >= early_stopper.patience:
                 torch.save(model.state_dict(), args.model_path)
                 print(f"Training finished. Model saved to {args.model_path}")
            print("Training completed.")

            # 3. Evaluation (only if training was done)
            if input("\nDo you want to evaluate the trained model? (yes/no): ").strip().lower() == 'yes':
                print("\nStep 3: Evaluating model...")
                if not os.path.exists(args.model_path):
                    print(f"Model file not found at {args.model_path}. Cannot evaluate.")
                    return
                
                # Reload metadata as it might have been overwritten or is needed fresh
                with open(meta_path, 'r') as f: meta = json.load(f)
                loaded_seq_length = meta.get('seq_length', args.seq_length)
                model_seq_len = loaded_seq_length

                test_inputs_path = os.path.join(args.processed_data_dir, 'test_inputs.npy')
                test_labels_path = os.path.join(args.processed_data_dir, 'test_labels.npy')
                if not os.path.exists(test_inputs_path) or not os.path.exists(test_labels_path):
                    print(f"Test data not found in {args.processed_data_dir}."); return

                test_inputs = np.load(test_inputs_path)
                test_labels_data = np.load(test_labels_path, allow_pickle=True) # Renamed

                test_loader = DataLoader(EvalDataset(test_inputs, test_labels_data), batch_size=args.batch_size, shuffle=False, collate_fn=eval_collate_fn)
                
                # Re-instantiate or load the model. For simplicity, re-instantiate with same args.
                # Ensure model architecture matches saved one. Best practice is to save config with model.
                eval_model = BERT4Rec(meta['num_items'], args.hidden_size, args.num_heads, args.num_layers, model_seq_len, args.dropout).to(device)
                try:
                    eval_model.load_state_dict(torch.load(args.model_path, map_location=device))
                except RuntimeError as e: 
                    print(f"Error loading model for evaluation: {e}"); return
                
                test_metrics = evaluate_rerank(eval_model, test_loader, device, meta['num_items'], args.k, args.neg_samples)
                print(f"Evaluation Result: Test Recall@{args.k}: {test_metrics['recall']:.4f}, Test NDCG@{args.k}: {test_metrics['ndcg']:.4f}")
                print("Evaluation completed.")
            else:
                print("Skipping evaluation.")
        else:
            print("Skipping training and subsequent steps.")
        print("\nInteractive mode finished.")
        
    elif args.action == "preprocess":
        print(f"Preprocessing data from {args.raw_data_path} and saving to {args.processed_data_dir} with sequence length {args.seq_length}...")
        # run_preprocess will internally call ensure_dataset_exists
        run_preprocess(data_path=args.raw_data_path, output_dir=args.processed_data_dir, seq_length_setting=args.seq_length)
    
    elif args.action == "train":
        print("Training model (direct action)...")
        # Ensure metadata is loaded correctly
        meta_path = os.path.join(args.processed_data_dir, 'metadata.json')
        if not os.path.exists(meta_path):
            print(f"Metadata file not found at {meta_path}. Please run preprocessing first.")
            return
        with open(meta_path, 'r') as f: meta = json.load(f)
        loaded_seq_length = meta.get('seq_length', args.seq_length)
        # Ensure consistency if seq_length from metadata is different from arg
        model_seq_len = loaded_seq_length 
        train_inputs = np.load(os.path.join(args.processed_data_dir, 'train_inputs.npy'))
        val_inputs = np.load(os.path.join(args.processed_data_dir, 'val_inputs.npy'))
        val_labels = np.load(os.path.join(args.processed_data_dir, 'val_labels.npy'), allow_pickle=True)
        mask_token_id = meta['num_items'] + 1
        train_loader = DataLoader(TrainDataset(train_inputs, mask_token_id, model_seq_len, args.mask_prob), batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(EvalDataset(val_inputs, val_labels), batch_size=args.batch_size, shuffle=False, collate_fn=eval_collate_fn)
        model = BERT4Rec(meta['num_items'], args.hidden_size, args.num_heads, args.num_layers, model_seq_len, args.dropout).to(device)
        optimizer = AdamW(model.parameters(), lr=args.lr)
        num_training_steps = args.epochs * len(train_loader)
        scheduler = CosineAnnealingLR(optimizer, T_max=num_training_steps, eta_min=args.lr*0.01)
        early_stopper = EarlyStopping(patience=args.early_stopping_patience)
        print(f"Starting training for {args.epochs} epochs...")
        for epoch in range(1, args.epochs + 1):
            model.train()
            epoch_loss = 0.0
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", leave=False)
            for input_ids, labels_batch in progress_bar: # Renamed
                input_ids, labels_batch = input_ids.to(device), labels_batch.to(device)
                optimizer.zero_grad()
                logits = model(input_ids)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels_batch.view(-1), ignore_index=-100)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                epoch_loss += loss.item()
                progress_bar.set_postfix({'loss': loss.item()})
            avg_epoch_loss = epoch_loss / len(train_loader)
            val_metrics = evaluate_rerank(model, val_loader, device, meta['num_items'], args.k, args.neg_samples)
            print(f"Epoch {epoch}: Avg Loss={avg_epoch_loss:.4f}, Val Recall@{args.k}={val_metrics['recall']:.4f}, Val NDCG@{args.k}={val_metrics['ndcg']:.4f}, LR={optimizer.param_groups[0]['lr']:.6f}")
            if early_stopper(val_metrics['ndcg'], epoch, model, args.model_path):
                print("Early stopping triggered."); break
        if not early_stopper.counter >= early_stopper.patience:
             torch.save(model.state_dict(), args.model_path)
             print(f"Training finished. Model saved to {args.model_path}")

    elif args.action == "evaluate":
        print("Evaluating model (direct action)...")
        meta_path = os.path.join(args.processed_data_dir, 'metadata.json')
        if not os.path.exists(meta_path): print(f"Metadata file not found at {meta_path}."); return
        if not os.path.exists(args.model_path): print(f"Model file not found at {args.model_path}."); return
        with open(meta_path, 'r') as f: meta = json.load(f)
        loaded_seq_length = meta.get('seq_length', args.seq_length)
        model_seq_len = loaded_seq_length
        test_inputs = np.load(os.path.join(args.processed_data_dir, 'test_inputs.npy'))
        test_labels_data = np.load(os.path.join(args.processed_data_dir, 'test_labels.npy'), allow_pickle=True) # Renamed
        test_loader = DataLoader(EvalDataset(test_inputs, test_labels_data), batch_size=args.batch_size, shuffle=False, collate_fn=eval_collate_fn)
        model = BERT4Rec(meta['num_items'], args.hidden_size, args.num_heads, args.num_layers, model_seq_len, args.dropout).to(device)
        print(f"Loading model from {args.model_path}")
        try: model.load_state_dict(torch.load(args.model_path, map_location=device))
        except RuntimeError as e: print(f"Error loading state_dict: {e}"); return
        model.to(device)
        test_metrics = evaluate_rerank(model, test_loader, device, meta['num_items'], args.k, args.neg_samples)
        print(f"Test Recall@{args.k}: {test_metrics['recall']:.4f}, Test NDCG@{args.k}: {test_metrics['ndcg']:.4f}")

    elif args.action == "ablation":
        run_ablation_study(args)
    # No explicit else for parser.print_help() needed if 'interactive' is a valid choice handled above
    # However, if action somehow becomes None and not caught by nargs='?', default=... then an explicit error or help might be good.
    # But argparse should handle unknown choices automatically.

if __name__ == "__main__":
    main() 