#!/usr/bin/env python3
"""
Demo script for Enhanced BERT4Rec
Tests the model with a small subset of data
"""

import torch
import numpy as np
from data_processor import create_data_loaders, MovieLensDataProcessor
from enhanced_model import EnhancedBERT4Rec
from trainer import EnhancedBERT4RecTrainer, MultiTaskLoss
from torch.utils.data import DataLoader, Dataset
import json

# Helper Dataset for overfitting test
class TinyDataset(Dataset):
    def __init__(self, data_batch, device):
        self.data_batch = {k: v.to(device) for k, v in data_batch.items()}

    def __len__(self):
        return 1 # Single batch

    def __getitem__(self, idx):
        return self.data_batch

def create_tiny_data_batch(metadata, seq_length, device):
    """Creates a single, tiny batch of data for the overfitting test."""
    # Use actual metadata to ensure consistency for num_tokens, etc.
    # These are placeholder values, try to use some actual values from a few users/items
    # from your dataset if possible, or ensure they are within valid ranges of your vocabularies.

    num_items = metadata['num_items']
    num_genres = metadata['num_genres']
    # num_users = metadata['num_users'] # user_id in batch should be < num_users
    # num_occupations = metadata['num_occupations'] # occupation_id in batch should be < num_occupations
    
    # Example: 1 sequence, batch size 1 for this test data
    batch = {
        'input_ids': torch.tensor([[10, 20, 30, 0, 0] + [i for i in range(1, seq_length - 4)]], dtype=torch.long), # Padded sequence
        'seq_genres': torch.randint(0, num_genres, (1, seq_length, metadata.get('max_genres', 5)), dtype=torch.long),
        'seq_years': torch.randint(1980, 2000, (1, seq_length), dtype=torch.long),
        'user_id': torch.tensor([1], dtype=torch.long), # Ensure user_id < num_users
        'user_gender': torch.tensor([1], dtype=torch.long), # 0, 1, or 2
        'user_age': torch.tensor([2], dtype=torch.long),    # Age group index
        'user_occupation': torch.tensor([3], dtype=torch.long), # Ensure occupation < num_occupations
        'input_timestamps': torch.randint(0, 365, (1, seq_length), dtype=torch.long), # Day of year
        'labels': torch.tensor([[-100, -100, 50, -100, -100] + [-100]*(seq_length-5)], dtype=torch.long), # MLM label, e.g., item 50 at index 2
        'target_item': torch.tensor([60], dtype=torch.long), # Next item to predict
        'target_rating': torch.tensor([4], dtype=torch.long) # Target rating (0-4 for 5 classes if using CrossEntropy)
    }
    # Ensure sequence lengths are consistent
    for k, v in batch.items():
        if k not in ['user_id', 'user_gender', 'user_age', 'user_occupation', 'target_item', 'target_rating']:
            if v.shape[1] != seq_length and len(v.shape) > 1:
                print(f"Adjusting shape for {k}. Original: {v.shape}")
                # This is a simplistic adjustment, be careful.
                # For overfitting, it's crucial that seq_length matches model's max_seq_len
                if k == 'input_ids' or k == 'labels':
                     current_seq = v[0].tolist()
                     batch[k] = torch.tensor([current_seq[:seq_length] + [0]*(seq_length - len(current_seq))], dtype=torch.long)
                elif k == 'seq_genres':
                    current_genres = v[0].tolist()
                    padded_genres = [g[:metadata.get('max_genres',5)] + [[0]*metadata.get('max_genres',5)]*(metadata.get('max_genres',5)-len(g)) if len(g) < metadata.get('max_genres',5) else g[:metadata.get('max_genres',5)] for g in current_genres]
                    batch[k] = torch.tensor([padded_genres[:seq_length] + [[0]*metadata.get('max_genres',5)]*(seq_length - len(padded_genres))], dtype=torch.long)

                else: # seq_years, input_timestamps
                    current_seq = v[0].tolist()
                    batch[k] = torch.tensor([current_seq[:seq_length] + [1990]*(seq_length - len(current_seq))], dtype=torch.long) # Pad with default year or timestamp
                print(f"Adjusted shape for {k} to {batch[k].shape}")


    return {k: v.to(device) for k, v in batch.items()}

def run_demo():
    """Run a quick demo with reduced parameters"""
    
    print("🚀 Enhanced BERT4Rec Demo")
    print("=" * 50)
    
    # Device selection
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Using device: {device}")
    
    # Demo configuration (reduced for quick testing)
    config = {
        # Model parameters (smaller for demo)
        'hidden_size': 128,
        'num_heads': 2,
        'num_layers': 2,
        'seq_length': 20,
        'dropout': 0.1,
        'genre_embed_size': 32, # Add and match default or make smaller if desired
        'user_embed_size': 64,  # Add and match default or make smaller if desired
        # 'temporal_embed_size': 32, # Explicitly set, ensure it matches hidden_size if added directly
        # For run_demo, let's ensure temporal_embed_size matches hidden_size if direct addition is used
        # This relies on the model's default behavior or direct addition logic.
        # If the model had a projection, this wouldn't be strictly necessary to match.
        
        # Training parameters (quick training)
        'batch_size': 16,
        'learning_rate': 1e-3,
        'max_epochs': 3,
        'patience': 5,
        'max_grad_norm': 1.0,
        
        # Multi-task weights
        'task_weights': {
            'item_prediction': 1.0,
            'next_item': 1.0,
            'rating_prediction': 0.3,
            'user_preference': 0.2
        },
        
        # Data parameters
        'mask_prob': 0.15,
        'train_ratio': 0.9,  # Use more data for training in demo
        
        # Evaluation
        'eval_every': 1,
        'save_every': 5,
        
        # Logging
        'use_wandb': False,
        'checkpoint_dir': 'demo_checkpoints'
    }
    
    print("📊 Creating data loaders...")
    try:
        # For demo, let's use a very small subset of data
        # Initialize processor directly to get all sequences first
        processor = MovieLensDataProcessor(
            data_path='ml-1m/',
            seq_length=config['seq_length']
        )
        all_sequences = processor.create_sequences(min_seq_length=5) # Get all available sequences
        
        # Use a small fraction for the demo
        demo_max_sequences = 500 # For example, use 500 sequences for the demo
        if len(all_sequences) > demo_max_sequences:
            import random
            random.shuffle(all_sequences) # Shuffle to get a diverse small set
            demo_sequences = all_sequences[:demo_max_sequences]
        else:
            demo_sequences = all_sequences
        
        print(f"Using {len(demo_sequences)} sequences for the demo (after sampling/limiting).")

        # Metadata needs to be compatible with what the model expects.
        # The original `create_data_loaders` returns metadata containing the processor.
        # We need to ensure the metadata here is sufficient.
        # Vocab sizes come from the processor instance.
        # For demo, it is important that num_items etc are from the *full* dataset so model init is correct.
        # Define metadata *before* creating datasets that depend on it.
        metadata = {
            'num_items': processor.item2idx.__len__(),
            'num_users': processor.user2idx.__len__(),
            'num_genres': processor.genre2idx.__len__(),
            'num_occupations': processor.occupation2idx.__len__(),
            'seq_length': config['seq_length'],
            'max_year': processor.movies_df['year'].max() if not processor.movies_df.empty else 2020,
            'min_year': processor.movies_df['year'].min() if not processor.movies_df.empty else 1900,
            'max_age_group': max(processor.age2group.values()) if processor.age2group else 7
        }

        # Split demo_sequences into train/val for EnhancedMovieLensDataset
        import numpy as np # ensure numpy is imported for shuffle if not already
        # np.random.shuffle(demo_sequences) # already shuffled with random.shuffle
        split_idx = int(len(demo_sequences) * config['train_ratio'])
        train_demo_sequences = demo_sequences[:split_idx]
        val_demo_sequences = demo_sequences[split_idx:]

        # Create datasets using the small subset
        from data_processor import EnhancedMovieLensDataset # Make sure this is imported
        train_dataset = EnhancedMovieLensDataset(train_demo_sequences, metadata['num_items'], config['mask_prob'])
        val_dataset = EnhancedMovieLensDataset(val_demo_sequences, metadata['num_items'], config['mask_prob'])

        train_loader = DataLoader(
            train_dataset, 
            batch_size=config['batch_size'], 
            shuffle=True,
            num_workers=0, # Use 0 for num_workers on CPU/MPS to avoid issues, esp. in demo
            pin_memory=False # Set pin_memory to False when num_workers is 0 or on CPU/MPS
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=config['batch_size'], 
            shuffle=False,
            num_workers=0,
            pin_memory=False
        )

        print(f"✅ Demo data loaded successfully!")
        print(f"   - Training batches: {len(train_loader)}")
        print(f"   - Validation batches: {len(val_loader)}")
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return False
    
    print("\n🤖 Initializing model...")
    try:
        model = EnhancedBERT4Rec(
            num_items=metadata['num_items'],
            num_genres=metadata['num_genres'],
            num_users=metadata['num_users'],
            num_occupations=metadata['num_occupations'],
            hidden_size=config['hidden_size'],
            num_heads=config['num_heads'],
            num_layers=config['num_layers'],
            max_seq_len=config['seq_length'],
            dropout=config['dropout'],
            max_year=metadata.get('max_year', 2020),
            min_year=metadata.get('min_year', 1930),
            max_age=metadata.get('max_age_group', 7) +1,
            genre_embed_size=config['genre_embed_size'],
            user_embed_size=config['user_embed_size'],
            temporal_embed_size=config['hidden_size'] # Ensure temporal matches hidden_size for demo
        )
        
        num_params = sum(p.numel() for p in model.parameters())
        print(f"✅ Model initialized successfully!")
        print(f"   - Parameters: {num_params:,}")
        print(f"   - Hidden size: {config['hidden_size']}")
        print(f"   - Attention heads: {config['num_heads']}")
        
    except Exception as e:
        print(f"❌ Error initializing model: {e}")
        return False
    
    print("\n🏋️ Starting training...")
    try:
        model = model.to(device) # Ensure model is on the correct device before trainer init
        trainer = EnhancedBERT4RecTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config,
            device=device # Pass the determined device to the trainer
        )
        
        print(f"✅ Trainer initialized!")
        print(f"   - Device: {trainer.device}")
        print(f"   - Optimizer: AdamW")
        print(f"   - Learning rate: {config['learning_rate']}")
        
        # Run training
        trainer.train(max_epochs=config['max_epochs'])
        
        print(f"✅ Training completed!")
        
    except Exception as e:
        print(f"❌ Error during training: {e}")
        return False
    
    print("\n🎯 Testing inference...")
    try:
        from inference import RecommendationEngine
        
        # Use the best model for inference
        engine = RecommendationEngine(
            model_path='demo_checkpoints/best_model.pt',
            metadata_path='metadata.json'
        )
        
        # Test recommendations for a sample user
        sample_sequence = [1, 2, 3, 5, 10]  # Example sequence
        recommendations = engine.get_recommendations(
            user_id=1,
            item_sequence=sample_sequence,
            k=5
        )
        
        print(f"✅ Inference successful!")
        print(f"   - Input sequence: {sample_sequence}")
        print(f"   - Top-5 recommendations:")
        
        for i, rec in enumerate(recommendations, 1):
            print(f"     {i}. Movie {rec['item_id']} (Score: {rec['score']:.4f})")
            print(f"        Genres: {', '.join(rec['genres']) if rec['genres'] else 'Unknown'}")
        
    except Exception as e:
        print(f"❌ Error during inference: {e}")
        print("   Note: This might be expected if training was very short")
        return False
    
    print("\n🎉 Demo completed successfully!")
    print("=" * 50)
    print("✨ Key achievements:")
    print("   - Data processing: ✅")
    print("   - Model initialization: ✅") 
    print("   - Training loop: ✅")
    print("   - Inference engine: ✅")
    print("\n💡 Next steps:")
    print("   - Run full training with python trainer.py")
    print("   - Experiment with different hyperparameters")
    print("   - Analyze model performance and attention weights")
    
    return True

def test_components():
    """Test individual components"""
    
    print("\n🔧 Testing individual components...")
    
    # Test data loading with minimal data
    print("1. Testing data processor...")
    try:
        processor = MovieLensDataProcessor('ml-1m/', seq_length=10)
        sequences = processor.create_sequences(min_seq_length=3)
        print(f"   ✅ Created {len(sequences)} sequences")
    except Exception as e:
        print(f"   ❌ Data processor error: {e}")
        return False
    
    # Test model forward pass
    print("2. Testing model forward pass...")
    try:
        model_for_loss_test = EnhancedBERT4Rec( # Renamed to avoid conflict if model is used later
            num_items=100, num_genres=20, num_users=50, num_occupations=10,
            hidden_size=64, num_heads=2, num_layers=1, max_seq_len=10,
            user_embed_size=32 # Explicitly set for clarity in test
        )
        
        # Create dummy batch
        batch_size, seq_len = 2, 10
        dummy_batch = {
            'input_ids': torch.randint(1, 100, (batch_size, seq_len)),
            'movie_genres': torch.randint(0, 20, (batch_size, seq_len, 5)),
            'movie_years': torch.randint(1990, 2020, (batch_size, seq_len)),
            'user_ids': torch.randint(1, 50, (batch_size,)),
            'user_genders': torch.randint(0, 3, (batch_size,)),
            'user_ages': torch.randint(0, 8, (batch_size,)),
            'user_occupations': torch.randint(1, 10, (batch_size,)),
        }
        
        with torch.no_grad():
            outputs_model_pass = model_for_loss_test(**dummy_batch) # Use new model name
            print(f"   ✅ Forward pass successful, output shape: {outputs_model_pass['item_logits'].shape}")
            
    except Exception as e:
        print(f"   ❌ Model forward pass error: {e}")
        return False
    
    print("3. Testing multi-task loss...")
    try:
        criterion = MultiTaskLoss()
        
        # Vocab size from the dummy model used for forward pass test
        # model_for_loss_test has num_items = 100, so vocab_size = 102
        # user_embed_size = 32 for this model instance
        item_vocab_size = model_for_loss_test.vocab_size 
        rating_classes = 5 # As defined in EnhancedBERT4Rec rating head
        user_embedding_size_for_test = model_for_loss_test.user_pref_head.out_features # Should be 32

        # Dummy outputs and targets
        batch_s, seq_l = 2, 10
        outputs_for_loss = {
            'item_logits': torch.randn(batch_s, seq_l, item_vocab_size),
            'rating_logits': torch.randn(batch_s, seq_l, rating_classes),
            'user_pref_logits': torch.randn(batch_s, seq_l, user_embedding_size_for_test)
        }
        
        # Generate valid labels: either -100 (ignore) or in [0, item_vocab_size-1]
        raw_labels = torch.randint(0, item_vocab_size, (batch_s, seq_l))
        mask_for_ignore = torch.rand(batch_s, seq_l) < 0.2 # ~20% ignored
        labels_for_loss = raw_labels.masked_fill(mask_for_ignore, -100)
        
        target_items_for_loss = torch.randint(0, item_vocab_size, (batch_s,)) # Target items for next_item task
        # Ensure no -100 if item_loss for next_item doesn't expect it for valid targets
        # item_loss (CrossEntropy) for next_item uses ignore_index=-100. So, if a target is -100, it's ignored.
        # For testing, better to have valid targets: 0 to item_vocab_size-1.
        target_items_for_loss = torch.clamp(target_items_for_loss, 0, item_vocab_size -1)

        target_ratings_for_loss = torch.randint(0, rating_classes, (batch_s,)) # Ratings 0-4 for 5 classes
        
        # User embeddings for user_preference_loss. Shape: [batch_size, user_embedding_size]
        user_embeddings_for_loss_target = torch.randn(batch_s, user_embedding_size_for_test)
        
        losses = criterion(
            outputs=outputs_for_loss, 
            labels=labels_for_loss, 
            target_items=target_items_for_loss, 
            target_ratings=target_ratings_for_loss, 
            user_embeddings=user_embeddings_for_loss_target
        )
        print(f"   ✅ Loss computation successful: total_loss={losses['total'].item():.4f}")
        assert 'item_prediction' in losses, "Item prediction loss missing"
        assert 'next_item' in losses, "Next item loss missing"
        assert 'rating_prediction' in losses, "Rating prediction loss missing"
        assert 'user_preference' in losses, "User preference loss missing"
        print(f"     Individual losses: item={losses['item_prediction'].item():.4f}, next_item={losses['next_item'].item():.4f}, rating={losses['rating_prediction'].item():.4f}, user_pref={losses['user_preference'].item():.4f}")

    except Exception as e:
        print(f"   ❌ Loss computation error: {e}")
        return False
    
    print("✅ All component tests passed!")
    return True

def run_overfitting_test():
    print("🧪 Starting Overfitting Test")
    print("-" * 50)
    # device = 'cuda' if torch.cuda.is_available() else 'cpu' # Old way
    # Improved device selection
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Using device: {device}")

    # First, we need metadata. Load it using DataProcessor
    # This ensures vocabs for genres, occupations etc. are available.
    print("Loading metadata via DataProcessor...")
    try:
        # Create a dummy processor instance just to load and process metadata
        # This will create/load metadata.json if it doesn't exist based on ml-1m
        # For the overfitting test, we rely on this metadata.json being present.
        data_proc = MovieLensDataProcessor(data_path='ml-1m/', seq_length=20)
        metadata_path = 'metadata.json' # Assuming data_proc saves it here or it exists
        if not torch.os.path.exists(metadata_path):
            print(f"Metadata file not found at {metadata_path}. Attempting to save it via DataProcessor.")
            data_proc.save_metadata(metadata_path) # Save it if it was just generated
            if not torch.os.path.exists(metadata_path):
                 print("Failed to create metadata.json. Overfitting test cannot proceed.")
                 return False
        
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        print("Metadata loaded successfully.")
    except Exception as e:
        print(f"Could not load or create metadata: {e}. Overfitting test aborted.")
        print("Ensure 'ml-1m/' directory is populated (e.g. by running trainer.py once or manually downloading).")
        return False

    overfit_seq_length = 10 # Keep it small for overfitting test

    # Overfitting configuration
    # Define hidden_size first to use it within the config dict for temporal_embed_size
    overfit_hidden_size = 32 

    config = {
        'hidden_size': overfit_hidden_size, # Small model
        'num_heads': 1,
        'num_layers': 1,
        'seq_length': overfit_seq_length, # Crucial: should match data
        'dropout': 0.0,   # CRITICAL: No dropout
        'learning_rate': 1e-3,
        'max_epochs': 300, # More epochs to ensure overfitting
        'weight_decay': 0.0, # CRITICAL: No weight decay
        'task_weights': {
            'item_prediction': 1.0,
            'next_item': 1.0, # Loss for this might be noisy if only one next item
            'rating_prediction': 1.0,
            'user_preference': 0.5
        },
        'batch_size': 1, # Single batch for overfitting
        'use_wandb': False,
        'checkpoint_dir': 'overfit_checkpoints', # Separate checkpoints
         # These need to be derived from the actual loaded metadata
        'num_items': metadata['num_items'],
        'num_genres': metadata['num_genres'],
        'num_users': metadata['num_users'],
        'num_occupations': metadata['num_occupations'],
        'max_year': metadata.get('max_year', 2005), # Ensure these exist in metadata or provide defaults
        'min_year': metadata.get('min_year', 1930),
        'max_age': metadata.get('max_age_group', 7) + 1, # metadata has age groups
        'genre_embed_size': 16,
        'user_embed_size': 16,
        'temporal_embed_size': overfit_hidden_size, # Match hidden_size for direct addition
    }
    
    # Create a single tiny batch for training and validation
    tiny_batch = create_tiny_data_batch(metadata, config['seq_length'], device)
    
    # Check if user_id and occupation_id in tiny_batch are valid
    if tiny_batch['user_id'].item() >= config['num_users']:
        print(f"Warning: user_id {tiny_batch['user_id'].item()} in tiny_batch is out of bounds for num_users {config['num_users']}. Clamping to 0.")
        tiny_batch['user_id'] = torch.tensor([0], dtype=torch.long, device=device)
    if tiny_batch['user_occupation'].item() >= config['num_occupations']:
        print(f"Warning: user_occupation {tiny_batch['user_occupation'].item()} in tiny_batch is out of bounds for num_occupations {config['num_occupations']}. Clamping to 0.")
        tiny_batch['user_occupation'] = torch.tensor([0], dtype=torch.long, device=device)


    tiny_dataset = TinyDataset(tiny_batch, device)
    # Use the same tiny dataset for train and val loader for overfitting test
    overfit_train_loader = DataLoader(tiny_dataset, batch_size=config['batch_size'], shuffle=False)
    overfit_val_loader = DataLoader(tiny_dataset, batch_size=config['batch_size'], shuffle=False)

    print("Initializing model for overfitting test...")
    model = EnhancedBERT4Rec(
        num_items=config['num_items'],
        num_genres=config['num_genres'],
        num_users=config['num_users'],
        num_occupations=config['num_occupations'],
        hidden_size=config['hidden_size'],
        num_heads=config['num_heads'],
        num_layers=config['num_layers'],
        max_seq_len=config['seq_length'], # Use seq_length from config
        dropout=config['dropout'],
        max_year=config['max_year'],
        min_year=config['min_year'],
        max_age=config['max_age'],
        genre_embed_size=config['genre_embed_size'],
        user_embed_size=config['user_embed_size'],
        temporal_embed_size=config['temporal_embed_size']
    ).to(device)

    criterion = MultiTaskLoss(config.get('task_weights', {}))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"]
    )

    print("Starting overfitting training loop...")
    min_loss_achieved = float('inf')
    overfit_success = False

    for epoch in range(1, config['max_epochs'] + 1):
        model.train()
        epoch_total_loss = 0.0
        
        # Since DataLoader wraps the single batch, this loop runs once.
        for batch_data in overfit_train_loader: # batch_data is our tiny_batch
            optimizer.zero_grad()
            
            # Correctly unpack the batch from DataLoader
            # Since tiny_dataset.__getitem__ returns the full batch dictionary,
            # and DataLoader batch_size=1, batch_data will have an extra leading dimension for each tensor.
            # We need to remove this outer batch dimension for each tensor before passing to the model.
            
            model_input_batch = {
                k: v[0] if isinstance(v, torch.Tensor) else v 
                for k, v in batch_data.items()
            }

            outputs = model(
                input_ids=model_input_batch['input_ids'],
                movie_genres=model_input_batch['seq_genres'],
                movie_years=model_input_batch['seq_years'],
                user_ids=model_input_batch['user_id'],
                user_genders=model_input_batch['user_gender'],
                user_ages=model_input_batch['user_age'],
                user_occupations=model_input_batch['user_occupation'],
                timestamps=model_input_batch['input_timestamps'],
                return_all_heads=True
            )
            
            user_embeddings_for_loss = model.user_embeddings(model_input_batch['user_id'])

            losses = criterion(
                outputs=outputs,
                labels=model_input_batch['labels'],
                target_items=model_input_batch['target_item'],
                target_ratings=model_input_batch['target_rating'],
                user_embeddings=user_embeddings_for_loss
            )
            
            losses['total'].backward()
            optimizer.step()
            epoch_total_loss += losses['total'].item()

            # Log individual losses
            log_msg = f"Epoch {epoch}/{config['max_epochs']}, Total Loss: {losses['total'].item():.6f}"
            for k_loss, v_loss in losses.items(): # renamed k to k_loss to avoid conflict
                if k_loss != 'total':
                    log_msg += f", {k_loss}: {v_loss.item():.4f}"
            
            # Calculate Recall@10 for the next item prediction
            recall_k = 10
            with torch.no_grad(): # Ensure no gradients for metric calculation
                # model.eval() # Not strictly needed if using outputs from training forward pass directly
                         # but good practice if we were re-running forward pass for eval.
                
                # item_logits shape: [batch_size, seq_len, vocab_size]
                # We need logits for the position corresponding to next_item prediction.
                # MultiTaskLoss uses outputs['item_logits'][:, -1, :] for 'next_item' loss.
                next_item_logits = outputs['item_logits'][:, -1, :] # Shape: (batch_size, vocab_size)
                true_target_item = model_input_batch['target_item']    # Shape: (batch_size)

                # Get top K predicted item indices
                # _, top_k_indices = torch.topk(next_item_logits, k=recall_k, dim=1)
                # The line above considers all vocab items, including PAD and MASK.
                # For a more focused recall, consider only valid item indices [1, num_items]
                # However, for simplicity in overfitting, predicting the raw index is fine.
                
                # Ensure true_target_item is on the same device as logits for comparison if not already
                # true_target_item = true_target_item.to(next_item_logits.device) 
                # model_input_batch should already be on the correct device.

                top_k_values, top_k_indices = torch.topk(next_item_logits, k=recall_k, dim=1)
                
                # Expand true_target_item to compare: (batch_size) -> (batch_size, 1)
                expanded_true_target = true_target_item.unsqueeze(1)
                
                # Check if true_target_item is in top_k_indices for each item in the batch
                # For our single item batch, this will be a tensor of shape [1]
                correct_predictions = torch.sum(top_k_indices == expanded_true_target, dim=1)
                recall_at_k_value = correct_predictions.float().mean().item() # Average recall over the batch (will be 0 or 1 for batch_size 1)
            
            log_msg += f", Recall@{recall_k}: {recall_at_k_value:.2f}"
            print(log_msg)
            # model.train() # Switch back if model.eval() was called

        if epoch_total_loss < min_loss_achieved:
            min_loss_achieved = epoch_total_loss
        
        if epoch_total_loss < 0.15: # Relaxed threshold
            print(f"🎉 Overfitting test successful! Loss ({epoch_total_loss:.6f}) is very small.")
            overfit_success = True
            break
        elif epoch % 20 == 0: # Print less frequently after initial epochs
             print(f"Epoch {epoch}, Current Min Loss: {min_loss_achieved:.6f}")


    if not overfit_success:
        print(f"⚠️ Overfitting test might not have fully succeeded. Minimum loss achieved: {min_loss_achieved:.6f}")
    
    print("-" * 50)
    return overfit_success

if __name__ == '__main__':
    # Test components first
    if test_components():
        print("\n" + "="*50)
        # Run overfitting test
        overfitting_passed = run_overfitting_test()
        print("\n" + "="*50)
        if overfitting_passed:
            print("✅ Overfitting test passed. Model can learn on a small batch.")
            print("Proceeding to full demo...")
            # Run full demo
            run_demo()
        else:
            print("❌ Overfitting test failed or did not converge sufficiently.")
            print("   Check model, data, and loss function implementation.")

    else:
        print("❌ Component tests failed. Please check your installation.") 