import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import time
import json
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import wandb
import os

from enhanced_model import EnhancedBERT4Rec
from data_processor import create_data_loaders


class MultiTaskLoss(nn.Module):
    """
    Multi-task loss with adaptive weighting
    """
    
    def __init__(self, task_weights: Dict[str, float] = None):
        super().__init__()
        
        # Default weights
        self.task_weights = task_weights or {
            'item_prediction': 1.0,
            'rating_prediction': 0.3,
            'user_preference': 0.2
        }
        
        # Loss functions
        self.item_loss = nn.CrossEntropyLoss(ignore_index=-100)
        self.rating_loss = nn.CrossEntropyLoss()
        self.user_pref_loss = nn.MSELoss()
    
    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        labels: torch.Tensor,
        target_items: torch.Tensor,
        target_ratings: torch.Tensor,
        user_embeddings: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Compute multi-task loss
        
        Args:
            outputs: Model outputs containing all heads
            labels: Masked language model labels
            target_items: Next item targets
            target_ratings: Rating targets
            user_embeddings: Ground truth user embeddings
        """
        losses = {}
        
        # Task 1: Masked item prediction (BERT-style)
        if 'item_logits' in outputs:
            item_logits = outputs['item_logits'].view(-1, outputs['item_logits'].size(-1))
            labels_flat = labels.view(-1)
            losses['item_prediction'] = self.item_loss(item_logits, labels_flat)
        
        # Task 2: Next item prediction
        if 'item_logits' in outputs:
            next_item_logits = outputs['item_logits'][:, -1, :]  # Last position
            losses['next_item'] = self.item_loss(next_item_logits, target_items)
        
        # Task 3: Rating prediction
        if 'rating_logits' in outputs:
            rating_logits = outputs['rating_logits'][:, -1, :]  # Last position
            losses['rating_prediction'] = self.rating_loss(rating_logits, target_ratings)
        
        # Task 4: User preference learning
        if 'user_pref_logits' in outputs:
            user_pref_pred = outputs['user_pref_logits'][:, -1, :]  # Last position
            losses['user_preference'] = self.user_pref_loss(user_pref_pred, user_embeddings)
        
        # Weighted total loss
        total_loss = 0
        for task, loss in losses.items():
            weight = self.task_weights.get(task, 1.0)
            total_loss += weight * loss
        
        losses['total'] = total_loss
        return losses


class EnhancedBERT4RecTrainer:
    """
    Trainer for Enhanced BERT4Rec with multi-task learning
    """
    
    def __init__(
        self,
        model: EnhancedBERT4Rec,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Dict,
        device: str
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        
        # Loss function
        self.criterion = MultiTaskLoss(config.get('task_weights', {}))
        
        # Optimizer
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=config.get('learning_rate', 1e-4),
            weight_decay=config.get('weight_decay', 0.01)
        )
        
        # Scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.get('max_epochs', 50),
            eta_min=config.get('min_learning_rate', 1e-6)
        )
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.train_losses = []
        self.val_losses = []
        
        # Create checkpoint directory
        self.checkpoint_dir = config.get('checkpoint_dir', 'checkpoints')
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # Initialize wandb if configured
        if config.get('use_wandb', False):
            wandb.init(
                project=config.get('wandb_project', 'enhanced-bert4rec'),
                name=config.get('experiment_name', 'experiment'),
                config=config
            )
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        epoch_losses = {
            'total': 0.0,
            'item_prediction': 0.0,
            'next_item': 0.0,
            'rating_prediction': 0.0,
            'user_preference': 0.0
        }
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f'Training Epoch {self.current_epoch + 1}')
        
        for batch in pbar:
            # Move batch to device
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            # Forward pass
            outputs = self.model(
                input_ids=batch['input_ids'],
                movie_genres=batch['seq_genres'],
                movie_years=batch['seq_years'],
                user_ids=batch['user_id'],
                user_genders=batch['user_gender'],
                user_ages=batch['user_age'],
                user_occupations=batch['user_occupation'],
                timestamps=batch['input_timestamps'],
                return_all_heads=True
            )
            
            # Get user embeddings for auxiliary task
            user_embeddings = self.model.user_embeddings(batch['user_id'])
            
            # Compute losses
            losses = self.criterion(
                outputs=outputs,
                labels=batch['labels'],
                target_items=batch['target_item'],
                target_ratings=batch['target_rating'],
                user_embeddings=user_embeddings
            )
            
            # Backward pass
            self.optimizer.zero_grad()
            losses['total'].backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), 
                max_norm=self.config.get('max_grad_norm', 1.0)
            )
            
            self.optimizer.step()
            
            # Accumulate losses
            for key, loss in losses.items():
                if key in epoch_losses:
                    epoch_losses[key] += loss.item()
            
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{losses['total'].item():.4f}",
                'lr': f"{self.optimizer.param_groups[0]['lr']:.2e}"
            })
        
        # Average losses
        for key in epoch_losses:
            epoch_losses[key] /= num_batches
        
        return epoch_losses
    
    def validate_epoch(self) -> Dict[str, float]:
        """Validate for one epoch"""
        self.model.eval()
        epoch_losses = {
            'total': 0.0,
            'item_prediction': 0.0,
            'next_item': 0.0,
            'rating_prediction': 0.0,
            'user_preference': 0.0
        }
        num_batches = 0
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc='Validation')
            
            for batch in pbar:
                # Move batch to device
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # Forward pass
                outputs = self.model(
                    input_ids=batch['input_ids'],
                    movie_genres=batch['seq_genres'],
                    movie_years=batch['seq_years'],
                    user_ids=batch['user_id'],
                    user_genders=batch['user_gender'],
                    user_ages=batch['user_age'],
                    user_occupations=batch['user_occupation'],
                    timestamps=batch['input_timestamps'],
                    return_all_heads=True
                )
                
                # Get user embeddings for auxiliary task
                user_embeddings = self.model.user_embeddings(batch['user_id'])
                
                # Compute losses
                losses = self.criterion(
                    outputs=outputs,
                    labels=batch['labels'],
                    target_items=batch['target_item'],
                    target_ratings=batch['target_rating'],
                    user_embeddings=user_embeddings
                )
                
                # Accumulate losses
                for key, loss in losses.items():
                    if key in epoch_losses:
                        epoch_losses[key] += loss.item()
                
                num_batches += 1
                
                # Update progress bar
                pbar.set_postfix({'loss': f"{losses['total'].item():.4f}"})
        
        # Average losses
        for key in epoch_losses:
            epoch_losses[key] /= num_batches
        
        return epoch_losses
    
    def compute_metrics(self, k_values: List[int] = [1, 5, 10, 20]) -> Dict[str, float]:
        """Compute Hit@K, NDCG@K, and Recall@K for given K values"""
        self.model.eval()
        
        all_model_predictions = [] 
        all_ground_truth_items = [] 
        
        with torch.no_grad():
            for batch_data_dict in tqdm(self.val_loader, desc="Evaluating", leave=False):
                current_batch_on_device = {
                    k: v.to(self.device) for k, v in batch_data_dict.items() if isinstance(v, torch.Tensor)
                }

                model_outputs = self.model(
                    input_ids=current_batch_on_device['input_ids'],
                    movie_genres=current_batch_on_device['seq_genres'],
                    movie_years=current_batch_on_device['seq_years'],
                    user_ids=current_batch_on_device['user_id'],
                    user_genders=current_batch_on_device['user_gender'],
                    user_ages=current_batch_on_device['user_age'],
                    user_occupations=current_batch_on_device['user_occupation'],
                    timestamps=current_batch_on_device['input_timestamps'],
                    return_all_heads=False 
                )
                
                preds_logits = model_outputs['item_logits'][:, -1, :]
                ground_truth_next_items = current_batch_on_device['target_item']
                
                for i in range(preds_logits.size(0)):
                    _, top_k_predicted_indices = torch.topk(preds_logits[i], max(k_values))
                    all_model_predictions.append(top_k_predicted_indices.cpu().numpy())
                    
                    actual_next_item_id = ground_truth_next_items[i].item()
                    all_ground_truth_items.append(actual_next_item_id)
        
        metrics = {}
        for k_val in k_values:
            hits_at_k = []
            ndcgs_at_k = []
            recalls_at_k = [] 
            
            for i in range(len(all_ground_truth_items)):
                top_k_preds_for_sample = all_model_predictions[i][:k_val] 
                label_for_sample = all_ground_truth_items[i]
                
                hit = 1 if label_for_sample in top_k_preds_for_sample else 0
                hits_at_k.append(hit)
                
                if hit:
                    rank_list = np.where(top_k_preds_for_sample == label_for_sample)[0]
                    if len(rank_list) > 0: # Check if label_for_sample was found
                        rank = rank_list[0] # rank is 0-indexed
                        ndcgs_at_k.append(1 / np.log2(rank + 2)) 
                    else: # Should ideally not happen if hit is 1, but as a safeguard
                        ndcgs_at_k.append(0)
                else:
                    ndcgs_at_k.append(0)

                recalls_at_k.append(hit)

            metrics[f'Hit@{k_val}'] = np.mean(hits_at_k)
            metrics[f'NDCG@{k_val}'] = np.mean(ndcgs_at_k)
            metrics[f'Recall@{k_val}'] = np.mean(recalls_at_k)
            
        return metrics
    
    def save_checkpoint(self, filename: str = None):
        """Save model checkpoint"""
        if filename is None:
            filename = f'checkpoint_epoch_{self.current_epoch}.pt'
        
        filepath = os.path.join(self.checkpoint_dir, filename)
        
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }
        
        torch.save(checkpoint, filepath)
        print(f"Checkpoint saved to {filepath}")
    
    def load_checkpoint(self, filepath: str):
        """Load model checkpoint"""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
        
        print(f"Checkpoint loaded from {filepath}")
    
    def train(self, max_epochs: int = None):
        """Main training loop"""
        max_epochs = max_epochs or self.config.get('max_epochs', 50)
        patience = self.config.get('patience', 10)
        early_stop_counter = 0
        
        print(f"Starting training for {max_epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        for epoch in range(self.current_epoch, max_epochs):
            self.current_epoch = epoch
            start_time = time.time()
            
            # Training
            train_losses = self.train_epoch()
            self.train_losses.append(train_losses)
            
            # Validation
            val_losses = self.validate_epoch()
            self.val_losses.append(val_losses)
            
            # Scheduler step
            self.scheduler.step()
            
            # Compute metrics periodically
            metrics = {}
            if epoch % self.config.get('eval_every', 5) == 0:
                metrics = self.compute_metrics()
            
            epoch_time = time.time() - start_time
            
            # Logging
            log_dict = {
                'epoch': epoch + 1,
                'train_loss': train_losses['total'],
                'val_loss': val_losses['total'],
                'lr': self.optimizer.param_groups[0]['lr'],
                'epoch_time': epoch_time,
                **{f'train_{k}': v for k, v in train_losses.items()},
                **{f'val_{k}': v for k, v in val_losses.items()},
                **metrics
            }
            
            # Print epoch summary
            print(f"\nEpoch {epoch + 1}/{max_epochs}:")
            print(f"  Train Loss: {train_losses['total']:.4f}")
            print(f"  Val Loss: {val_losses['total']:.4f}")
            if metrics:
                print(f"  Hit@10: {metrics.get('Hit@10', 0):.4f}")
                print(f"  NDCG@10: {metrics.get('NDCG@10', 0):.4f}")
                print(f"  Recall@10: {metrics.get('Recall@10', 0):.4f}")
            print(f"  Time: {epoch_time:.2f}s")
            
            # Log to wandb
            if self.config.get('use_wandb', False):
                wandb.log(log_dict)
            
            # Save best model
            if val_losses['total'] < self.best_val_loss:
                self.best_val_loss = val_losses['total']
                self.save_checkpoint('best_model.pt')
                early_stop_counter = 0
                print("  New best model saved!")
            else:
                early_stop_counter += 1
            
            # Early stopping
            if early_stop_counter >= patience:
                print(f"\nEarly stopping triggered after {patience} epochs without improvement")
                break
            
            # Save regular checkpoint
            if epoch % self.config.get('save_every', 10) == 0:
                self.save_checkpoint()
        
        print("Training completed!")


def main():
    """Main training script"""
    
    # Basic configuration (can be replaced with argparse or a config file)
    config = {
        'data_dir': 'ml-1m',
        'batch_size': 64,
        'max_seq_len': 50,
        'learning_rate': 1e-3,
        'max_epochs': 100,
        'patience': 15, # Early stopping patience
        'hidden_size': 48,
        'num_heads': 2,
        'num_layers': 2,
        'dropout': 0.2,
        'genre_embed_size': 16,
        'user_embed_size': 128,
        'temporal_embed_size': 256,
        'task_weights': {
            'item_prediction': 1.0,
            'next_item': 1.0, 
            'rating_prediction': 0.3,
            'user_preference': 0.2
        },
        'use_wandb': False, # Set to True to use WandB
        'checkpoint_dir': 'checkpoints',
        'metadata_path': 'metadata.json',
        'min_learning_rate': 1e-6, # For CosineAnnealingLR
        'max_grad_norm': 1.0, # For gradient clipping
        'num_workers': 0 # Number of workers for DataLoader
    }

    # Improved device selection
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # Load metadata
    print(f"Loading metadata from {config['metadata_path']}...")
    with open(config['metadata_path'], 'r') as f:
        metadata = json.load(f)
    print("Metadata loaded successfully.")

    # Create data loaders
    print("Creating data loaders...")
    train_loader, val_loader, _, metadata = create_data_loaders(
        data_dir=config['data_dir'],
        metadata=metadata,
        seq_length=config['max_seq_len'],
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        device=device # Pass device to create_data_loaders
    )
    print("Data loaders created.")

    # Initialize model
    print("Initializing model...")
    model = EnhancedBERT4Rec(
        num_items=metadata['num_items'],
        num_genres=metadata['num_genres'],
        num_users=metadata['num_users'],
        num_occupations=metadata['num_occupations'],
        hidden_size=config['hidden_size'],
        num_heads=config['num_heads'],
        num_layers=config['num_layers'],
        max_seq_len=config['max_seq_len'],
        dropout=config['dropout'],
        max_year=metadata.get('max_year', 2000),
        min_year=metadata.get('min_year', 1930),
        max_age=metadata.get('max_age', 56),
        genre_embed_size=config['genre_embed_size'],
        user_embed_size=config['user_embed_size'],
        temporal_embed_size=config['temporal_embed_size']
    ).to(device)
    print("Model initialized.")

    # Initialize trainer
    print("Initializing trainer...")
    trainer = EnhancedBERT4RecTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device # Pass the selected device
    )
    print("Trainer initialized.")
    
    # Start training
    print("Starting training...")
    trainer.train(max_epochs=config['max_epochs'])


if __name__ == '__main__':
    main() 