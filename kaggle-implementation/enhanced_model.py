import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from typing import Dict, List, Optional, Tuple


class EnhancedBERT4Rec(nn.Module):
    """
    Enhanced BERT4Rec with Multi-Head Architecture and Side Information
    
    Features:
    - Multi-task learning (next-item prediction + rating prediction)
    - Movie metadata integration (genres, year)
    - User demographics integration (age, gender, occupation)
    - Temporal embeddings for timestamps
    - Cross-attention between sequence and metadata
    """
    
    def __init__(
        self,
        num_items: int,
        num_genres: int,
        num_users: int,
        num_occupations: int,
        hidden_size: int = 256,
        num_heads: int = 4,
        num_layers: int = 4,
        max_seq_len: int = 50,
        dropout: float = 0.2,
        max_year: int = 2000,
        min_year: int = 1930,
        max_age: int = 56,
        genre_embed_size: int = 32,
        user_embed_size: int = 64,
        temporal_embed_size: int = 32
    ):
        super().__init__()
        
        self.pad_token = 0
        self.mask_token = num_items + 1
        self.vocab_size = num_items + 2
        self.hidden_size = hidden_size
        self.num_items = num_items
        
        # Core item embeddings
        self.item_embeddings = nn.Embedding(
            self.vocab_size, 
            hidden_size, 
            padding_idx=self.pad_token
        )
        
        # Positional embeddings (sinusoidal)
        self.register_buffer(
            'position_embeddings',
            self._get_sinusoidal_encoding(max_seq_len, hidden_size)
        )
        
        # Movie metadata embeddings
        self.genre_embeddings = nn.Embedding(num_genres, genre_embed_size)
        self.year_embeddings = nn.Embedding(max_year - min_year + 1, temporal_embed_size)
        
        # User embeddings
        self.user_embeddings = nn.Embedding(num_users, user_embed_size)
        self.gender_embeddings = nn.Embedding(3, 16)  # M, F, Unknown
        self.age_embeddings = nn.Embedding(8, 16)  # Age groups
        self.occupation_embeddings = nn.Embedding(num_occupations, 32)
        
        # Timestamp embeddings for temporal patterns
        self.timestamp_embeddings = nn.Embedding(366, temporal_embed_size)  # Day of year
        
        # Projection layer for temporal features
        self.temporal_projection = nn.Linear(temporal_embed_size, hidden_size)
        
        # Projection layers for side information
        self.movie_meta_projection = nn.Linear(
            genre_embed_size + temporal_embed_size,  # genres + year
            hidden_size
        )
        
        self.user_meta_projection = nn.Linear(
            user_embed_size + 16 + 16 + 32,  # user + gender + age + occupation
            hidden_size
        )
        
        # Cross-attention for metadata integration
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Multi-head outputs
        self.layer_norm = nn.LayerNorm(hidden_size)
        
        # Head 1: Next-item prediction (main task)
        self.item_prediction_head = nn.Linear(hidden_size, self.vocab_size)
        
        # Head 2: Rating prediction (auxiliary task)
        self.rating_prediction_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 5)  # 1-5 star ratings
        )
        
        # Head 3: User preference prediction (auxiliary task)
        self.user_pref_head = nn.Linear(hidden_size, user_embed_size)
        
        self.dropout = nn.Dropout(dropout)
        self.min_year = min_year
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with Xavier/Glorot initialization"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.xavier_uniform_(module.weight)
                if hasattr(module, 'padding_idx') and module.padding_idx is not None:
                    with torch.no_grad():
                        module.weight[module.padding_idx].fill_(0)
    
    def _get_sinusoidal_encoding(self, max_len: int, d_model: int) -> torch.Tensor:
        """Generate sinusoidal positional encodings"""
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)
    
    def _create_padding_mask(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Create padding mask for transformer"""
        return (input_ids == self.pad_token)
    
    def encode_movie_metadata(
        self, 
        item_ids: torch.Tensor,
        movie_genres: torch.Tensor,
        movie_years: torch.Tensor
    ) -> torch.Tensor:
        """
        Encode movie metadata
        
        Args:
            item_ids: [batch_size, seq_len]
            movie_genres: [batch_size, seq_len, max_genres]
            movie_years: [batch_size, seq_len]
        """
        batch_size, seq_len = item_ids.shape
        
        # Genre embeddings - aggregate multiple genres per movie
        genre_embeds = self.genre_embeddings(movie_genres)  # [B, S, max_genres, genre_dim]
        genre_embeds = genre_embeds.mean(dim=2)  # Average pool genres [B, S, genre_dim]
        
        # Year embeddings
        year_indices = (movie_years - self.min_year).clamp(0, self.year_embeddings.num_embeddings - 1)
        year_embeds = self.year_embeddings(year_indices)  # [B, S, temporal_dim]
        
        # Concatenate and project
        movie_meta = torch.cat([genre_embeds, year_embeds], dim=-1)  # [B, S, genre_dim + temporal_dim]
        movie_meta_proj = self.movie_meta_projection(movie_meta)  # [B, S, hidden_size]
        
        return movie_meta_proj
    
    def encode_user_metadata(
        self,
        user_ids: torch.Tensor,
        user_genders: torch.Tensor,
        user_ages: torch.Tensor,
        user_occupations: torch.Tensor
    ) -> torch.Tensor:
        """
        Encode user metadata
        
        Args:
            user_ids: [batch_size]
            user_genders: [batch_size]
            user_ages: [batch_size]  
            user_occupations: [batch_size]
        """
        user_embeds = self.user_embeddings(user_ids)
        gender_embeds = self.gender_embeddings(user_genders)
        age_embeds = self.age_embeddings(user_ages)
        occupation_embeds = self.occupation_embeddings(user_occupations)
        
        # Concatenate all user features
        user_meta = torch.cat([
            user_embeds, gender_embeds, age_embeds, occupation_embeds
        ], dim=-1)
        
        user_meta_proj = self.user_meta_projection(user_meta)  # [batch_size, hidden_size]
        
        return user_meta_proj
    
    def encode_temporal_features(self, timestamps: torch.Tensor) -> torch.Tensor:
        """
        Encode temporal features from timestamps
        
        Args:
            timestamps: [batch_size, seq_len] - Unix timestamps
        """
        # Convert to day of year (simplified temporal feature)
        # In practice, you might want more sophisticated temporal features
        days_of_year = (timestamps % (365 * 24 * 3600)) // (24 * 3600)
        days_of_year = days_of_year.clamp(0, 365)
        
        temporal_embeds = self.timestamp_embeddings(days_of_year.long())
        return temporal_embeds
    
    def forward(
        self,
        input_ids: torch.Tensor,
        movie_genres: torch.Tensor,
        movie_years: torch.Tensor,
        user_ids: torch.Tensor,
        user_genders: torch.Tensor,
        user_ages: torch.Tensor,
        user_occupations: torch.Tensor,
        timestamps: Optional[torch.Tensor] = None,
        return_all_heads: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with multi-head outputs
        
        Args:
            input_ids: [batch_size, seq_len]
            movie_genres: [batch_size, seq_len, max_genres]
            movie_years: [batch_size, seq_len]
            user_ids: [batch_size]
            user_genders: [batch_size]
            user_ages: [batch_size]
            user_occupations: [batch_size]
            timestamps: [batch_size, seq_len] (optional)
            return_all_heads: Whether to return all prediction heads
        """
        batch_size, seq_len = input_ids.shape
        
        # Core item embeddings + positional
        item_embeds = self.item_embeddings(input_ids)
        pos_embeds = self.position_embeddings[:, :seq_len, :]
        sequence_embeds = item_embeds + pos_embeds
        
        # Temporal embeddings (if available)
        if timestamps is not None:
            temporal_embeds = self.encode_temporal_features(timestamps)
            temporal_embeds_proj = self.temporal_projection(temporal_embeds)
            sequence_embeds = sequence_embeds + temporal_embeds_proj
        
        # Movie metadata embeddings
        movie_meta_embeds = self.encode_movie_metadata(input_ids, movie_genres, movie_years)
        
        # User metadata embeddings
        user_meta_embeds = self.encode_user_metadata(
            user_ids, user_genders, user_ages, user_occupations
        )
        
        # Combine sequence with movie metadata
        enhanced_sequence = sequence_embeds + movie_meta_embeds
        
        # Cross-attention with user metadata
        user_meta_expanded = user_meta_embeds.unsqueeze(1).expand(-1, seq_len, -1)
        
        # Apply cross-attention between sequence and user context
        attended_sequence, _ = self.cross_attention(
            query=enhanced_sequence,
            key=user_meta_expanded,
            value=user_meta_expanded,
            key_padding_mask=None
        )
        
        # Combine attended sequence with original
        combined_sequence = enhanced_sequence + attended_sequence
        combined_sequence = self.dropout(combined_sequence)
        
        # Create attention mask
        padding_mask = self._create_padding_mask(input_ids)
        
        # Transformer encoding
        hidden_states = self.transformer(
            combined_sequence,
            src_key_padding_mask=padding_mask
        )
        
        hidden_states = self.layer_norm(hidden_states)
        
        # Multi-head predictions
        outputs = {}
        
        # Head 1: Item prediction (main task)
        item_logits = self.item_prediction_head(hidden_states)
        outputs['item_logits'] = item_logits
        
        if return_all_heads:
            # Head 2: Rating prediction
            rating_logits = self.rating_prediction_head(hidden_states)
            outputs['rating_logits'] = rating_logits
            
            # Head 3: User preference prediction
            user_pref_logits = self.user_pref_head(hidden_states)
            outputs['user_pref_logits'] = user_pref_logits
            
            # Return hidden states for analysis
            outputs['hidden_states'] = hidden_states
        
        return outputs
    
    def predict_next_items(
        self,
        input_ids: torch.Tensor,
        movie_genres: torch.Tensor,
        movie_years: torch.Tensor,
        user_ids: torch.Tensor,
        user_genders: torch.Tensor,
        user_ages: torch.Tensor,
        user_occupations: torch.Tensor,
        timestamps: Optional[torch.Tensor] = None,
        k: int = 10
    ) -> torch.Tensor:
        """
        Predict next k items for recommendation
        
        Returns:
            top_k_items: [batch_size, k] - Top-k recommended item indices
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(
                input_ids, movie_genres, movie_years,
                user_ids, user_genders, user_ages, user_occupations,
                timestamps
            )
            
            # Get logits for last position (next item prediction)
            last_hidden = outputs['item_logits'][:, -1, :]  # [batch_size, vocab_size]
            
            # Remove special tokens from prediction
            last_hidden[:, self.pad_token] = float('-inf')
            last_hidden[:, self.mask_token] = float('-inf')
            
            # Get top-k predictions
            top_k_scores, top_k_items = torch.topk(last_hidden, k, dim=-1)
            
        return top_k_items 