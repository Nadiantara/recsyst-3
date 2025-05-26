import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import json
import re
from typing import Dict, List, Tuple, Optional
from collections import defaultdict


class MovieLensDataProcessor:
    """
    Enhanced data processor for MovieLens 1M with metadata
    """
    
    def __init__(self, data_path: str = "ml-1m/", seq_length: int = 50):
        self.data_path = data_path
        self.seq_length = seq_length
        self.max_genres = 5  # Maximum number of genres per movie
        
        # Mappings
        self.item2idx = {}
        self.idx2item = {}
        self.user2idx = {}
        self.idx2user = {}
        self.genre2idx = {}
        self.occupation2idx = {}
        self.year2idx = {}
        self.idx2year = {}
        self.num_unique_years = 0
        
        # Movie metadata
        self.movie_genres = {}  # movie_id -> list of genre indices
        self.movie_years = {}   # movie_id -> year
        
        # User metadata
        self.user_demographics = {}  # user_id -> {gender, age, occupation}
        
        self.pad_token = 0
        self.load_and_process_data()
    
    def load_and_process_data(self):
        """Load and process all MovieLens data"""
        print("Loading MovieLens 1M dataset...")
        
        # Load movies data
        self._load_movies()
        
        # Load users data
        self._load_users()
        
        # Load ratings data
        self._load_ratings()
        
        print(f"Dataset loaded:")
        print(f"- Users: {len(self.user2idx)}")
        print(f"- Items: {len(self.item2idx)}")
        print(f"- Genres: {len(self.genre2idx)}")
        print(f"- Occupations: {len(self.occupation2idx)}")
        print(f"- Total interactions: {len(self.ratings_df)}")
    
    def _load_movies(self):
        """Load movie metadata"""
        movies_file = f"{self.data_path}/movies.dat"
        
        movies_data = []
        with open(movies_file, 'r', encoding='iso-8859-1') as f:
            for line in f:
                parts = line.strip().split("::")
                movie_id = int(parts[0])
                title = parts[1]
                genres = parts[2].split("|")
                
                # Extract year from title
                year_match = re.search(r'\((\d{4})\)', title)
                year = int(year_match.group(1)) if year_match else 1995  # Default year
                
                movies_data.append({
                    'movie_id': movie_id,
                    'title': title,
                    'genres': genres,
                    'year': year
                })
        
        self.movies_df = pd.DataFrame(movies_data)
        
        # Build genre vocabulary
        all_genres = set()
        for genres in self.movies_df['genres']:
            all_genres.update(genres)
        
        self.genre2idx = {genre: idx + 1 for idx, genre in enumerate(sorted(all_genres))}
        self.genre2idx['[PAD]'] = 0  # Padding for genres
        
        # Build item vocabulary (add 1 to leave 0 for padding)
        unique_movies = sorted(self.movies_df['movie_id'].unique())
        self.item2idx = {movie_id: idx + 1 for idx, movie_id in enumerate(unique_movies)}
        self.idx2item = {idx: movie_id for movie_id, idx in self.item2idx.items()}
        
        # Build year vocabulary
        unique_years = sorted(self.movies_df['year'].unique())
        self.year2idx = {year: idx for idx, year in enumerate(unique_years)}
        self.idx2year = {idx: year for year, idx in self.year2idx.items()}
        self.num_unique_years = len(unique_years)
        self.default_year_idx = self.year2idx.get(1995, 0) # Default year index for padding
        
        # Process movie metadata
        for _, row in self.movies_df.iterrows():
            movie_id = row['movie_id']
            
            # Genre indices (padded to max_genres)
            genre_indices = [self.genre2idx[g] for g in row['genres']]
            while len(genre_indices) < self.max_genres:
                genre_indices.append(0)  # Pad with 0
            self.movie_genres[movie_id] = genre_indices[:self.max_genres]
            
            # Year (store as 0-indexed ID)
            self.movie_years[movie_id] = self.year2idx[row['year']]
        
        print(f"Loaded {len(self.movies_df)} movies")
    
    def _load_users(self):
        """Load user metadata"""
        users_file = f"{self.data_path}/users.dat"
        
        users_data = []
        with open(users_file, 'r', encoding='iso-8859-1') as f:
            for line in f:
                parts = line.strip().split("::")
                user_id = int(parts[0])
                gender = parts[1]
                age = int(parts[2])
                occupation = int(parts[3])
                zip_code = parts[4]
                
                users_data.append({
                    'user_id': user_id,
                    'gender': gender,
                    'age': age,
                    'occupation': occupation,
                    'zip_code': zip_code
                })
        
        self.users_df = pd.DataFrame(users_data)
        
        # Build user vocabulary
        unique_users = sorted(self.users_df['user_id'].unique())
        self.user2idx = {user_id: idx + 1 for idx, user_id in enumerate(unique_users)}
        self.idx2user = {idx: user_id for user_id, idx in self.user2idx.items()}
        
        # Build occupation vocabulary
        unique_occupations = sorted(self.users_df['occupation'].unique())
        self.occupation2idx = {occ: idx + 1 for idx, occ in enumerate(unique_occupations)}
        
        # Gender mapping
        self.gender2idx = {'M': 1, 'F': 2, 'U': 0}  # U for unknown/padding
        
        # Age group mapping (simplified)
        self.age2group = {
            1: 0, 18: 1, 25: 2, 35: 3, 45: 4, 50: 5, 56: 6, 60: 7
        }
        
        # Process user demographics
        for _, row in self.users_df.iterrows():
            user_id = row['user_id']
            
            # Map age to age group
            age_group = 0
            for age_threshold in sorted(self.age2group.keys()):
                if row['age'] >= age_threshold:
                    age_group = self.age2group[age_threshold]
            
            self.user_demographics[user_id] = {
                'gender': self.gender2idx.get(row['gender'], 0),
                'age': age_group,
                'occupation': self.occupation2idx[row['occupation']]
            }
        
        print(f"Loaded {len(self.users_df)} users")
    
    def _load_ratings(self):
        """Load ratings data"""
        ratings_file = f"{self.data_path}/ratings.dat"
        
        ratings_data = []
        with open(ratings_file, 'r', encoding='iso-8859-1') as f:
            for line in f:
                parts = line.strip().split("::")
                user_id = int(parts[0])
                movie_id = int(parts[1])
                rating = int(parts[2])
                timestamp = int(parts[3])
                
                # Only include items and users that we have metadata for
                if movie_id in self.item2idx and user_id in self.user2idx:
                    ratings_data.append({
                        'user_id': user_id,
                        'movie_id': movie_id,
                        'rating': rating,
                        'timestamp': timestamp
                    })
        
        self.ratings_df = pd.DataFrame(ratings_data)
        
        # Sort by user and timestamp
        self.ratings_df = self.ratings_df.sort_values(['user_id', 'timestamp'])
        
        print(f"Loaded {len(self.ratings_df)} ratings")
    
    def _convert_to_json_serializable(self, data):
        """Recursively convert numpy types and non-string keys to JSON serializable formats."""
        if isinstance(data, dict):
            new_dict = {}
            for k, v in data.items():
                key = str(k) # Ensure all keys are strings
                new_dict[key] = self._convert_to_json_serializable(v)
            return new_dict
        elif isinstance(data, list):
            return [self._convert_to_json_serializable(item) for item in data]
        elif isinstance(data, np.integer):
            return int(data)
        elif isinstance(data, np.floating):
            return float(data)
        elif isinstance(data, np.ndarray):
            return self._convert_to_json_serializable(data.tolist())
        # Add other numpy types if necessary, e.g. np.bool_
        return data

    def save_metadata(self, save_path: str):
        """Save processed metadata to a JSON file"""
        print(f"Saving metadata to {save_path}...")
        
        # Prepare data for JSON serialization
        # Using the recursive _convert_to_json_serializable helper for robustness
        data_to_collect = {
            'item2idx': self.item2idx,
            'idx2item': self.idx2item,
            'user2idx': self.user2idx,
            'idx2user': self.idx2user,
            'genre2idx': self.genre2idx,
            'idx2genre': {idx: name for name, idx in self.genre2idx.items() if name != '[PAD]'}, # Create idx2genre, ensure idx is str later
            'occupation2idx': self.occupation2idx,
            'idx2occupation': {idx: name for name, idx in self.occupation2idx.items()}, # Create idx2occupation, ensure idx is str later
            'year2idx': self.year2idx,
            'idx2year': self.idx2year,
            
            'movie_genres': self.movie_genres,
            'movie_years': self.movie_years,
            'user_demographics': self.user_demographics,
            
            'pad_token': self.pad_token,
            'seq_length': self.seq_length,
            'max_genres': self.max_genres,
            
            'num_items': len(self.item2idx),
            'num_users': len(self.user2idx),
            'num_genres': len(self.genre2idx),
            'num_occupations': len(self.occupation2idx),
            'num_years': self.num_unique_years,
            'num_genders': len(self.gender2idx),
            'num_age_groups': len(set(self.age2group.values())) if self.age2group else 0,
            
            'min_year_val': self.movies_df['year'].min() if not self.movies_df.empty else 1900,
            'max_year_val': self.movies_df['year'].max() if not self.movies_df.empty else 2020,
            'max_age_group': max(self.age2group.values()) if self.age2group else 7 
        }

        # Convert the entire collected data using the recursive helper
        metadata_to_save = self._convert_to_json_serializable(data_to_collect)

        # Special handling for idx2genre and idx2occupation keys after conversion if they were int initially
        if 'idx2genre' in metadata_to_save:
            metadata_to_save['idx2genre'] = {str(k): v for k, v in metadata_to_save['idx2genre'].items()}
        if 'idx2occupation' in metadata_to_save:
            metadata_to_save['idx2occupation'] = {str(k): v for k, v in metadata_to_save['idx2occupation'].items()}

        try:
            with open(save_path, 'w') as f:
                json.dump(metadata_to_save, f, indent=4)
            print(f"✅ Metadata saved successfully to {save_path}")
        except TypeError as e:
            print(f"❌ Error saving metadata to JSON: {e}")
            print("   This usually means some data types were not JSON serializable.")
            print("   Problematic data structure might be:")
            # Try to print parts of the structure that might be causing issues
            # for k,v_dict in metadata_to_save.items():
            #     if isinstance(v_dict, dict):
            #         for sub_k, sub_v in v_dict.items():
            #             print(f"Key: {sub_k}, Type: {type(sub_k)}, Value type: {type(sub_v)}")
            #             if isinstance(sub_v, dict) or isinstance(sub_v, list):
            #                 pass # too verbose
            #             else:
            #                 print(f"    Value: {sub_v}")
            #             break # print one example per sub-dict
            #     break # print one example

    def create_sequences(self, min_seq_length: int = 5) -> Dict[str, List]:
        """
        Create sequences for training
        
        Args:
            min_seq_length: Minimum sequence length to include
            
        Returns:
            Dictionary with sequences and metadata
        """
        print("Creating sequences...")
        
        sequences = []
        user_sequences = defaultdict(list)
        
        # Group by user
        for _, row in self.ratings_df.iterrows():
            user_id = row['user_id']
            movie_id = row['movie_id']
            rating = row['rating']
            timestamp = row['timestamp']
            
            user_sequences[user_id].append({
                'movie_id': movie_id,
                'rating': rating,
                'timestamp': timestamp
            })
        
        # Create training sequences
        for user_id, user_items in user_sequences.items():
            if len(user_items) < min_seq_length:
                continue
            
            # Convert to indices
            item_indices = [self.item2idx[item['movie_id']] for item in user_items]
            ratings = [item['rating'] for item in user_items]
            timestamps = [item['timestamp'] for item in user_items]
            movie_ids = [item['movie_id'] for item in user_items]
            
            # Create overlapping sequences
            for i in range(len(item_indices) - 1):
                # Input sequence
                start_idx = max(0, i + 1 - self.seq_length)
                input_seq = item_indices[start_idx:i + 1]
                input_ratings = ratings[start_idx:i + 1]
                input_timestamps = timestamps[start_idx:i + 1]
                input_movie_ids = movie_ids[start_idx:i + 1]
                
                # Target (next item)
                target_item = item_indices[i + 1]
                target_rating = ratings[i + 1]
                target_movie_id = movie_ids[i + 1]
                
                # Pad sequences
                while len(input_seq) < self.seq_length:
                    input_seq.insert(0, self.pad_token)
                    input_ratings.insert(0, 0)
                    input_timestamps.insert(0, 0)
                    input_movie_ids.insert(0, 0)  # Use 0 as padding movie_id
                
                # Get movie metadata for sequence
                seq_genres = []
                seq_years = []
                
                for mid in input_movie_ids:
                    if mid == 0:  # Padding
                        seq_genres.append([0] * self.max_genres)
                        seq_years.append(self.default_year_idx) # Use default_year_idx for padding
                    else:
                        seq_genres.append(self.movie_genres[mid])
                        seq_years.append(self.movie_years[mid]) # Already 0-indexed
                
                # Get target movie metadata
                # Ensure target_year is also an index if it's used by the model directly
                # For now, model uses seq_years for temporal embeddings. If target_year metadata is needed, convert it.
                target_genres = self.movie_genres[target_movie_id] 
                # target_year = self.movie_years[target_movie_id] # This would be the indexed year

                user_demo = self.user_demographics[user_id]
                
                sequences.append({
                    'user_id': self.user2idx[user_id],
                    'input_ids': input_seq,
                    'target_item': target_item,
                    'input_ratings': input_ratings,
                    'target_rating': target_rating,
                    'input_timestamps': input_timestamps,
                    'seq_genres': seq_genres,
                    'seq_years': seq_years, # Now contains 0-indexed year IDs
                    # 'target_genres': target_genres, # Target metadata, if needed by loss/model
                    # 'target_year': target_year,     # Target metadata, if needed by loss/model
                    'user_gender': user_demo['gender'],
                    'user_age': user_demo['age'],
                    'user_occupation': user_demo['occupation']
                })
        
        print(f"Created {len(sequences)} training sequences")
        return sequences


class EnhancedMovieLensDataset(Dataset):
    """
    Enhanced dataset for BERT4Rec with metadata
    """
    
    def __init__(self, sequences: List[Dict], num_items: int, mask_prob: float = 0.15):
        self.sequences = sequences
        self.mask_prob = mask_prob
        self.num_items = num_items # Store num_items from full dataset
        
        # Get vocab size from first sequence (THIS IS PROBLEMATIC - see below)
        # if sequences: # Old logic
        #     sample_seq = sequences[0]
        #     self.vocab_size = max(max(sample_seq['input_ids']), sample_seq['target_item']) + 2
        #     self.mask_token = self.vocab_size - 1

        # Correct way to define vocab_size and mask_token based on full dataset num_items
        # Consistent with EnhancedBERT4Rec model: vocab_size = num_items + 2 (item, PAD, MASK)
        self.pad_token_id = 0 # Assuming PAD is 0
        self.mask_token_id = self.num_items + 1 # MASK is num_items + 1
        self.actual_vocab_size = self.num_items + 2 # Total size for embedding layer
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        
        # Create masked sequence for BERT-style training
        input_ids = sequence['input_ids'].copy()
        labels = [-100] * len(input_ids)  # -100 = ignore in loss
        
        # Random masking
        for i in range(len(input_ids)):
            if input_ids[i] != 0 and np.random.random() < self.mask_prob:
                labels[i] = input_ids[i]
                
                prob = np.random.random()
                if prob < 0.8:
                    input_ids[i] = self.mask_token_id  # 80% mask
                elif prob < 0.9:
                    # 10% random item from valid item range [1, num_items]
                    input_ids[i] = np.random.randint(1, self.num_items + 1)
                # 10% keep original
        
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.long),
            'target_item': torch.tensor(sequence['target_item'], dtype=torch.long),
            'target_rating': torch.tensor(sequence['target_rating'] - 1, dtype=torch.long),  # 0-indexed
            'user_id': torch.tensor(sequence['user_id'] - 1, dtype=torch.long), # Shift to be 0-indexed for nn.Embedding
            'user_gender': torch.tensor(sequence['user_gender'], dtype=torch.long),
            'user_age': torch.tensor(sequence['user_age'], dtype=torch.long),
            'user_occupation': torch.tensor(sequence['user_occupation'] - 1, dtype=torch.long), # Shift to be 0-indexed for nn.Embedding
            'seq_genres': torch.tensor(sequence['seq_genres'], dtype=torch.long),
            'seq_years': torch.tensor(sequence['seq_years'], dtype=torch.long),
            'input_timestamps': torch.tensor(sequence['input_timestamps'], dtype=torch.long),
            'input_ratings': torch.tensor(sequence['input_ratings'], dtype=torch.long)
        }


def create_data_loaders(
    data_dir: str = "ml-1m/", # Renamed from data_path for consistency
    metadata: Optional[Dict] = None, # Added metadata parameter
    processor: Optional[MovieLensDataProcessor] = None, # Optional processor parameter
    seq_length: int = 50,
    train_ratio: float = 0.8,
    batch_size: int = 32,
    mask_prob: float = 0.15,
    num_workers: int = 4, # Added num_workers
    device: str = 'cpu'   # Added device parameter
) -> Tuple[DataLoader, DataLoader, Optional[DataLoader], Dict]: # Test loader is optional, added Dict for metadata
    """
    Create data loaders for training, validation, and testing.
    If metadata is provided, it uses it. Otherwise, it initializes a processor.
    Returns the data loaders and the (potentially updated) metadata dictionary.
    """
    
    processor_instance = processor # Use passed-in processor if available
    current_metadata = metadata # Use passed-in metadata if available

    if not processor_instance:
        print("No processor provided. Initializing MovieLensDataProcessor.")
        processor_instance = MovieLensDataProcessor(data_path=data_dir, seq_length=seq_length)
        print("Saving metadata from newly initialized processor.")
        processor_instance.save_metadata("metadata.json")
        # Since we just saved it, this is the definitive version to use.
        print("Loading metadata from newly saved file.")
        with open("metadata.json", 'r') as f:
            current_metadata = json.load(f)
    else:
        # Processor was provided. The passed 'current_metadata' might be from an old file
        # or None. We need to ensure it's consistent with the processor or load if None.
        if current_metadata is None:
            print("Processor provided, but no metadata. Attempting to load metadata.json.")
            try:
                with open("metadata.json", 'r') as f:
                    current_metadata = json.load(f)
                print("Loaded metadata.json successfully.")
            except FileNotFoundError:
                print("metadata.json not found. Will create it from the provided processor.")
                processor_instance.save_metadata("metadata.json")
                with open("metadata.json", 'r') as f:
                    current_metadata = json.load(f)
                print("Saved and reloaded metadata from provided processor.")
    
    if processor_instance is None:
        raise RuntimeError("Critical error: MovieLensDataProcessor instance could not be obtained or created.")
    if current_metadata is None:
        # This should ideally not be reached if the above logic is correct, 
        # but as a fallback, create metadata from processor.
        print("Fallback: metadata is still None. Creating and saving from processor_instance.")
        processor_instance.save_metadata("metadata.json")
        with open("metadata.json", 'r') as f:
            current_metadata = json.load(f)

    # Consolidate and verify required metadata fields from the processor instance (source of truth)
    required_keys_and_sources = {
        'num_items': lambda p: len(p.item2idx),
        'num_users': lambda p: len(p.user2idx),
        'num_genres': lambda p: len(p.genre2idx),
        'num_occupations': lambda p: len(p.occupation2idx),
        'num_years': lambda p: p.num_unique_years,
        'num_genders': lambda p: len(p.gender2idx),
        'num_age_groups': lambda p: len(set(p.age2group.values())) if p.age2group else 0,
        'seq_length': lambda p: p.seq_length,
        'pad_token': lambda p: p.pad_token,
        'max_genres': lambda p: p.max_genres,
        # Mappings that might be useful (though model primarily needs counts for embedding sizes)
        # 'item2idx': lambda p: p.item2idx, 
        # 'year2idx': lambda p: p.year2idx,
    }

    for key, getter in required_keys_and_sources.items():
        processor_value = getter(processor_instance)
        if key not in current_metadata or current_metadata[key] != processor_value:
            if key in current_metadata and current_metadata[key] != processor_value:
                print(f"Metadata Fix: Key '{key}' mismatch. Disk/Passed='{current_metadata[key]}', Processor='{processor_value}'. Using processor's.")
            elif key not in current_metadata:
                print(f"Metadata Fix: Key '{key}' missing. Using processor's value '{processor_value}'.")
            current_metadata[key] = processor_value

    num_items_for_dataset = current_metadata['num_items']
    train_sequences, val_sequences, test_sequences = [], [], []

    # Try to use pre-loaded sequences from current_metadata first
    if 'train_sequences' in current_metadata and 'val_sequences' in current_metadata:
        print("Using pre-loaded sequences from metadata.")
        train_sequences = current_metadata['train_sequences']
        val_sequences = current_metadata['val_sequences']
        test_sequences = current_metadata.get('test_sequences', [])
    else:
        print("Creating sequences using MovieLensDataProcessor...")
        all_raw_sequences_list = processor_instance.create_sequences()
        
        if not all_raw_sequences_list:
            raise ValueError(
                "MovieLensDataProcessor generated an empty list of sequences. "
                "Check data processing logic, data files (ml-1m folder), ratings, and min_seq_length."
            )
        print(f"Generated {len(all_raw_sequences_list)} raw sequences.")

        user_grouped_sequences = defaultdict(list)
        for seq_data in all_raw_sequences_list:
            user_grouped_sequences[seq_data['user_id']].append(seq_data)

        for user_id_key, sequences_for_user in user_grouped_sequences.items():
            if not sequences_for_user: continue
            n_user_sequences = len(sequences_for_user)
            if n_user_sequences == 1 and train_ratio < 1.0:
                train_sequences.extend(sequences_for_user)
                continue
            split_idx_val = int(n_user_sequences * train_ratio)
            if train_ratio > 0 and split_idx_val == 0 and n_user_sequences > 0:
                 split_idx_val = 1 
            user_train_seqs = sequences_for_user[:split_idx_val]
            user_val_seqs = sequences_for_user[split_idx_val:]
            train_sequences.extend(user_train_seqs)
            if user_val_seqs:
                val_sequences.extend(user_val_seqs)
        
        print(f"Total sequences after split: Train {len(train_sequences)}, Val {len(val_sequences)}")
        if not train_sequences:
            print("Warning: No training sequences were created after split. Check train_ratio and data.")
        if not val_sequences and train_ratio < 1.0:
             print("Warning: No validation sequences were created after split. Check train_ratio and data.")

    if num_items_for_dataset == 0:
        raise ValueError("Could not determine 'num_items' for dataset creation.")

    train_dataset = EnhancedMovieLensDataset(train_sequences, num_items=num_items_for_dataset, mask_prob=mask_prob)
    val_dataset = EnhancedMovieLensDataset(val_sequences, num_items=num_items_for_dataset, mask_prob=mask_prob)
    test_dataset = None
    if test_sequences:
        test_dataset = EnhancedMovieLensDataset(test_sequences, num_items=num_items_for_dataset, mask_prob=mask_prob)

    pin_memory = (device == 'cuda')
    if device == 'mps' and pin_memory:
        print("Warning: 'pin_memory' is True but MPS does not support it well. Setting to False.")
        pin_memory = False

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory
    )
    test_loader = None
    if test_dataset and len(test_dataset) > 0:
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory
        )
    
    return train_loader, val_loader, test_loader, current_metadata # Return updated metadata


if __name__ == "__main__":
    # Example usage
    data_dir = "ml-1m/"
    metadata = None
    processor = None
    seq_length = 50
    train_ratio = 0.8
    batch_size = 32
    mask_prob = 0.15
    num_workers = 4
    device = 'cpu'

    train_loader, val_loader, test_loader, metadata = create_data_loaders(
        data_dir=data_dir,
        metadata=metadata,
        processor=processor,
        seq_length=seq_length,
        train_ratio=train_ratio,
        batch_size=batch_size,
        mask_prob=mask_prob,
        num_workers=num_workers,
        device=device
    )

    print("Data loaders created successfully.")
    print(f"Train loader: {train_loader}")
    print(f"Val loader: {val_loader}")
    print(f"Test loader: {test_loader}")
    print(f"Metadata: {metadata}") 