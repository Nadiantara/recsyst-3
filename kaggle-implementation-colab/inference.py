import torch
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import argparse

from enhanced_model import EnhancedBERT4Rec
from data_processor import MovieLensDataProcessor


class RecommendationEngine:
    """
    Inference engine for Enhanced BERT4Rec recommendations
    """
    
    def __init__(
        self,
        model_path: str,
        metadata_path: str,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.device = device
        
        # Load metadata
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
        
        # Load model weights and config from checkpoint
        checkpoint = torch.load(model_path, map_location=device)
        model_config = checkpoint.get('config')

        if model_config is None:
            raise ValueError("'config' not found in checkpoint. Cannot initialize model.")

        # Initialize model using parameters from loaded config and metadata
        self.model = EnhancedBERT4Rec(
            num_items=self.metadata['num_items'],
            num_genres=self.metadata['num_genres'], 
            num_users=self.metadata['num_users'],
            num_occupations=self.metadata['num_occupations'],
            
            hidden_size=model_config.get('hidden_size', 256),
            num_heads=model_config.get('num_heads', 4),
            num_layers=model_config.get('num_layers', 4),
            max_seq_len=model_config.get('seq_length', self.metadata.get('seq_length', 50)),
            dropout=model_config.get('dropout', 0.2),
            
            genre_embed_size=model_config.get('genre_embed_size', 32),
            user_embed_size=model_config.get('user_embed_size', 64),
            temporal_embed_size=model_config.get('temporal_embed_size', model_config.get('hidden_size', 32)),

            max_year=self.metadata.get('max_year', 2020),
            min_year=self.metadata.get('min_year', 1930),
            # Ensure max_age is derived correctly if 'max_age_group' is in metadata
            max_age=self.metadata.get('max_age_group', self.metadata.get('max_age', 56)) + 1 
        )
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(device)
        self.model.eval()
        
        # Create reverse mappings
        self.idx2item = {int(v): int(k) for k, v in self.metadata['item2idx'].items()}
        self.idx2user = {int(v): int(k) for k, v in self.metadata['user2idx'].items()}
        
        print(f"Model loaded successfully!")
        print(f"Device: {device}")
        print(f"Vocabulary size: {self.metadata['num_items']} items")
    
    def prepare_sequence_data(
        self,
        user_id: int,
        item_sequence: List[int],
        timestamps: Optional[List[int]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Prepare sequence data for inference
        
        Args:
            user_id: User ID
            item_sequence: List of item IDs in chronological order
            timestamps: Optional timestamps for each item
            
        Returns:
            Dictionary with tensors ready for model input
        """
        seq_length = self.metadata['seq_length']
        max_genres = self.metadata['max_genres']
        
        # Convert item IDs to indices
        item_indices = []
        for item_id in item_sequence:
            if str(item_id) in self.metadata['item2idx']:
                item_indices.append(self.metadata['item2idx'][str(item_id)])
            else:
                print(f"Warning: Item {item_id} not found in vocabulary")
                item_indices.append(0)  # Use padding token
        
        # Pad sequence
        if len(item_indices) > seq_length:
            item_indices = item_indices[-seq_length:]  # Take last seq_length items
        else:
            # Pad with zeros at the beginning
            padding_length = seq_length - len(item_indices)
            item_indices = [0] * padding_length + item_indices
        
        # Get movie metadata for sequence
        seq_genres = []
        seq_years = []
        
        for idx in item_indices:
            if idx == 0:  # Padding
                seq_genres.append([0] * max_genres)
                seq_years.append(1995)  # Default year
            else:
                # Get original movie ID
                movie_id = self.idx2item[idx]
                
                # Get genres (padded to max_genres)
                if str(movie_id) in self.metadata['movie_genres']:
                    genres = self.metadata['movie_genres'][str(movie_id)]
                    seq_genres.append(genres)
                else:
                    seq_genres.append([0] * max_genres)
                
                # Get year
                if str(movie_id) in self.metadata['movie_years']:
                    year = self.metadata['movie_years'][str(movie_id)]
                    seq_years.append(year)
                else:
                    seq_years.append(1995)
        
        # Get user metadata
        user_demo = self.metadata['user_demographics'].get(str(user_id), {
            'gender': 0, 'age': 0, 'occupation': 1
        })
        
        # Prepare timestamps
        if timestamps is None:
            timestamps = [0] * seq_length
        else:
            if len(timestamps) > seq_length:
                timestamps = timestamps[-seq_length:]
            else:
                padding_length = seq_length - len(timestamps)
                timestamps = [0] * padding_length + timestamps
        
        # Convert to tensors
        data = {
            'input_ids': torch.tensor([item_indices], dtype=torch.long, device=self.device),
            'movie_genres': torch.tensor([seq_genres], dtype=torch.long, device=self.device),
            'movie_years': torch.tensor([seq_years], dtype=torch.long, device=self.device),
            'user_ids': torch.tensor([self.metadata['user2idx'].get(str(user_id), 1)], 
                                   dtype=torch.long, device=self.device),
            'user_genders': torch.tensor([user_demo['gender']], dtype=torch.long, device=self.device),
            'user_ages': torch.tensor([user_demo['age']], dtype=torch.long, device=self.device),
            'user_occupations': torch.tensor([user_demo['occupation']], dtype=torch.long, device=self.device),
            'timestamps': torch.tensor([timestamps], dtype=torch.long, device=self.device)
        }
        
        return data
    
    def get_recommendations(
        self,
        user_id: int,
        item_sequence: List[int],
        k: int = 10,
        timestamps: Optional[List[int]] = None,
        filter_seen: bool = True
    ) -> List[Dict]:
        """
        Get top-k recommendations for a user
        
        Args:
            user_id: User ID
            item_sequence: User's interaction history (item IDs)
            k: Number of recommendations to return
            timestamps: Optional timestamps for each item
            filter_seen: Whether to filter out items the user has already seen
            
        Returns:
            List of recommendation dictionaries with item info and scores
        """
        # Prepare input data
        data = self.prepare_sequence_data(user_id, item_sequence, timestamps)
        
        with torch.no_grad():
            # Get model predictions
            predictions = self.model.predict_next_items(
                input_ids=data['input_ids'],
                movie_genres=data['movie_genres'],
                movie_years=data['movie_years'],
                user_ids=data['user_ids'],
                user_genders=data['user_genders'],
                user_ages=data['user_ages'],
                user_occupations=data['user_occupations'],
                timestamps=data['timestamps'],
                k=k * 2 if filter_seen else k  # Get more to filter
            )
            
            # Get prediction scores for ranking
            outputs = self.model(
                input_ids=data['input_ids'],
                movie_genres=data['movie_genres'],
                movie_years=data['movie_years'],
                user_ids=data['user_ids'],
                user_genders=data['user_genders'],
                user_ages=data['user_ages'],
                user_occupations=data['user_occupations'],
                timestamps=data['timestamps']
            )
            
            # Get scores for last position
            scores = torch.softmax(outputs['item_logits'][0, -1, :], dim=0)
        
        # Convert predictions to item IDs
        pred_indices = predictions[0].cpu().numpy()
        recommendations = []
        
        for idx in pred_indices:
            if idx in self.idx2item:
                item_id = self.idx2item[idx]
                
                # Filter seen items if requested
                if filter_seen and item_id in item_sequence:
                    continue
                
                # Get item metadata
                item_info = {
                    'item_id': item_id,
                    'score': float(scores[idx].cpu()),
                    'genres': [],
                    'year': None,
                    'title': f"Movie {item_id}"  # Placeholder
                }
                
                # Add genre information
                if str(item_id) in self.metadata['movie_genres']:
                    genre_indices = self.metadata['movie_genres'][str(item_id)]
                    idx2genre = {v: k for k, v in self.metadata['genre2idx'].items()}
                    item_info['genres'] = [idx2genre.get(g, 'Unknown') for g in genre_indices if g > 0]
                
                # Add year information
                if str(item_id) in self.metadata['movie_years']:
                    item_info['year'] = self.metadata['movie_years'][str(item_id)]
                
                recommendations.append(item_info)
                
                if len(recommendations) >= k:
                    break
        
        return recommendations
    
    def get_user_profile(self, user_id: int) -> Dict:
        """
        Get user profile information
        
        Args:
            user_id: User ID
            
        Returns:
            Dictionary with user demographics
        """
        user_demo = self.metadata['user_demographics'].get(str(user_id), {})
        
        # Convert indices back to readable format
        gender_map = {v: k for k, v in self.metadata['gender2idx'].items()}
        age_groups = ['Under 18', '18-24', '25-34', '35-44', '45-49', '50-55', '56+', '60+']
        
        profile = {
            'user_id': user_id,
            'gender': gender_map.get(user_demo.get('gender', 0), 'Unknown'),
            'age_group': age_groups[user_demo.get('age', 0)] if user_demo.get('age', 0) < len(age_groups) else 'Unknown',
            'occupation_code': user_demo.get('occupation', 0)
        }
        
        return profile
    
    def explain_recommendation(
        self,
        user_id: int,
        item_sequence: List[int],
        recommended_item_id: int
    ) -> Dict:
        """
        Provide explanation for a recommendation
        
        Args:
            user_id: User ID
            item_sequence: User's interaction history
            recommended_item_id: ID of recommended item
            
        Returns:
            Dictionary with explanation
        """
        # Get user profile
        user_profile = self.get_user_profile(user_id)
        
        # Get recommended item info
        item_genres = []
        if str(recommended_item_id) in self.metadata['movie_genres']:
            genre_indices = self.metadata['movie_genres'][str(recommended_item_id)]
            idx2genre = {v: k for k, v in self.metadata['genre2idx'].items()}
            item_genres = [idx2genre.get(g, 'Unknown') for g in genre_indices if g > 0]
        
        item_year = self.metadata['movie_years'].get(str(recommended_item_id), 'Unknown')
        
        # Analyze user's genre preferences
        user_genre_counts = {}
        for item_id in item_sequence[-10:]:  # Last 10 items
            if str(item_id) in self.metadata['movie_genres']:
                genre_indices = self.metadata['movie_genres'][str(item_id)]
                idx2genre = {v: k for k, v in self.metadata['genre2idx'].items()}
                for g_idx in genre_indices:
                    if g_idx > 0:
                        genre = idx2genre.get(g_idx, 'Unknown')
                        user_genre_counts[genre] = user_genre_counts.get(genre, 0) + 1
        
        # Find genre overlap
        preferred_genres = sorted(user_genre_counts.items(), key=lambda x: x[1], reverse=True)[:3]
        genre_overlap = [g for g in item_genres if g in user_genre_counts]
        
        explanation = {
            'recommended_item': {
                'item_id': recommended_item_id,
                'genres': item_genres,
                'year': item_year
            },
            'user_profile': user_profile,
            'user_preferred_genres': [g[0] for g in preferred_genres],
            'genre_overlap': genre_overlap,
            'recent_sequence_length': len(item_sequence),
            'explanation_text': self._generate_explanation_text(
                user_profile, preferred_genres, item_genres, genre_overlap, item_year
            )
        }
        
        return explanation
    
    def _generate_explanation_text(
        self,
        user_profile: Dict,
        preferred_genres: List[Tuple],
        item_genres: List[str],
        genre_overlap: List[str],
        item_year: int
    ) -> str:
        """Generate human-readable explanation text"""
        
        explanation_parts = []
        
        # User profile based explanation
        explanation_parts.append(
            f"Based on your profile ({user_profile['gender']}, {user_profile['age_group']})"
        )
        
        # Genre preference explanation
        if preferred_genres:
            top_genres = [g[0] for g in preferred_genres[:2]]
            explanation_parts.append(
                f"and your preference for {' and '.join(top_genres)} movies"
            )
        
        # Genre overlap explanation
        if genre_overlap:
            explanation_parts.append(
                f"this {' and '.join(genre_overlap)} movie from {item_year} matches your taste"
            )
        else:
            explanation_parts.append(
                f"this {' and '.join(item_genres)} movie from {item_year} offers something new for you"
            )
        
        return ", ".join(explanation_parts) + "."


def main():
    """Demo script for the recommendation engine"""
    parser = argparse.ArgumentParser(description='Enhanced BERT4Rec Recommendation Engine')
    parser.add_argument('--model_path', required=True, help='Path to trained model checkpoint')
    parser.add_argument('--metadata_path', required=True, help='Path to metadata.json')
    parser.add_argument('--user_id', type=int, default=1, help='User ID for recommendations')
    parser.add_argument('--k', type=int, default=10, help='Number of recommendations')
    
    args = parser.parse_args()
    
    # Initialize recommendation engine
    print("Loading recommendation engine...")
    engine = RecommendationEngine(args.model_path, args.metadata_path)
    
    # Example user sequence (you would get this from user's interaction history)
    example_sequence = [1, 2, 3, 5, 10, 15, 20, 25, 30]  # Example movie IDs
    
    print(f"\nGetting recommendations for User {args.user_id}...")
    print(f"User's recent interactions: {example_sequence}")
    
    # Get recommendations
    recommendations = engine.get_recommendations(
        user_id=args.user_id,
        item_sequence=example_sequence,
        k=args.k
    )
    
    print(f"\nTop {args.k} Recommendations:")
    print("-" * 60)
    
    for i, rec in enumerate(recommendations, 1):
        print(f"{i:2d}. Movie {rec['item_id']:4d} | Score: {rec['score']:.4f}")
        print(f"     Genres: {', '.join(rec['genres']) if rec['genres'] else 'Unknown'}")
        print(f"     Year: {rec['year']}")
        print()
    
    # Get explanation for first recommendation
    if recommendations:
        print("Explanation for first recommendation:")
        print("-" * 60)
        explanation = engine.explain_recommendation(
            user_id=args.user_id,
            item_sequence=example_sequence,
            recommended_item_id=recommendations[0]['item_id']
        )
        print(explanation['explanation_text'])
        print()
        print(f"Genre overlap: {explanation['genre_overlap']}")
        print(f"User's preferred genres: {explanation['user_preferred_genres']}")


if __name__ == '__main__':
    main() 