"""
Scaled Dot-Product Attention Implementation

This module implements the scaled dot-product attention mechanism as described
in "Attention Is All You Need" (Vaswani et al., 2017).

The attention function can be described as mapping a query and a set of key-value
pairs to an output, where the query, keys, values, and output are all vectors.
The output is computed as a weighted sum of the values, where the weight assigned
to each value is computed by a compatibility function of the query with the
corresponding key.

Mathematical formulation:
    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V

Where:
    - Q: Query matrix of shape (batch_size, seq_len_q, d_k)
    - K: Key matrix of shape (batch_size, seq_len_k, d_k)
    - V: Value matrix of shape (batch_size, seq_len_v, d_v)
    - d_k: Dimension of the key vectors
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot-Product Attention module.
    
    This implementation follows the original transformer paper and includes:
    - Proper scaling by sqrt(d_k) to prevent gradient vanishing
    - Support for attention masking
    - Optional dropout for regularization
    - Numerical stability optimizations
    
    Args:
        dropout (float): Dropout probability applied to attention weights.
                        Default: 0.0 (no dropout)
    """
    
    def __init__(self, dropout: float = 0.0):
        super(ScaledDotProductAttention, self).__init__()
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply scaled dot-product attention.
        
        Args:
            query (torch.Tensor): Query tensor of shape (batch_size, seq_len_q, d_k)
            key (torch.Tensor): Key tensor of shape (batch_size, seq_len_k, d_k)
            value (torch.Tensor): Value tensor of shape (batch_size, seq_len_v, d_v)
            mask (torch.Tensor, optional): Attention mask of shape 
                                         (batch_size, seq_len_q, seq_len_k)
                                         True values indicate positions to attend to,
                                         False values will be masked out.
        
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: 
                - output: Attention output of shape (batch_size, seq_len_q, d_v)
                - attention_weights: Attention weights of shape (batch_size, seq_len_q, seq_len_k)
        
        Raises:
            AssertionError: If input dimensions are incompatible
        """
        # Validate input shapes
        batch_size_q, seq_len_q, d_k = query.shape
        batch_size_k, seq_len_k, d_k_key = key.shape
        batch_size_v, seq_len_v, d_v = value.shape
        
        assert batch_size_q == batch_size_k == batch_size_v, \
            f"Batch sizes must match: {batch_size_q}, {batch_size_k}, {batch_size_v}"
        assert d_k == d_k_key, \
            f"Query and key dimensions must match: {d_k} vs {d_k_key}"
        assert seq_len_k == seq_len_v, \
            f"Key and value sequence lengths must match: {seq_len_k} vs {seq_len_v}"
        
        # Step 1: Compute attention scores
        # QK^T with shape (batch_size, seq_len_q, seq_len_k)
        scores = torch.matmul(query, key.transpose(-2, -1))
        
        # Step 2: Scale by sqrt(d_k) to prevent gradient vanishing
        # This is crucial for numerical stability with large dimensions
        scaling_factor = math.sqrt(d_k)
        scores = scores / scaling_factor
        
        # Step 3: Apply mask if provided
        if mask is not None:
            # Validate mask shape
            expected_mask_shape = (batch_size_q, seq_len_q, seq_len_k)
            assert mask.shape == expected_mask_shape, \
                f"Mask shape {mask.shape} doesn't match expected {expected_mask_shape}"
            
            # Apply mask by setting masked positions to large negative value
            # This ensures they become ~0 after softmax
            scores = scores.masked_fill(~mask, float('-inf'))
        
        # Step 4: Apply softmax to get attention weights
        # Use dim=-1 to normalize across the key dimension
        attention_weights = F.softmax(scores, dim=-1)
        
        # Handle case where entire row is masked (results in NaN from softmax)
        # Replace NaN with zeros
        attention_weights = torch.where(
            torch.isnan(attention_weights),
            torch.zeros_like(attention_weights),
            attention_weights
        )
        
        # Step 5: Apply dropout to attention weights (for regularization)
        # Note: Dropout is only applied during training
        attention_weights = self.dropout(attention_weights)
        
        # Step 6: Compute weighted sum of values
        # Matrix multiplication: (batch_size, seq_len_q, seq_len_k) @ (batch_size, seq_len_k, d_v)
        # Result shape: (batch_size, seq_len_q, d_v)
        output = torch.matmul(attention_weights, value)
        
        return output, attention_weights
    
    def extra_repr(self) -> str:
        """Return extra representation string for debugging."""
        return f'dropout={self.dropout.p}'


# Utility functions for attention mechanism
def create_causal_mask(seq_length: int, device: torch.device = None) -> torch.Tensor:
    """
    Create a causal (lower triangular) attention mask.
    
    This mask ensures that position i can only attend to positions j <= i,
    which is essential for autoregressive language modeling.
    
    Args:
        seq_length (int): Length of the sequence
        device (torch.device, optional): Device to place the mask on
    
    Returns:
        torch.Tensor: Boolean mask of shape (seq_length, seq_length)
                     True indicates positions that can be attended to
    """
    mask = torch.tril(torch.ones(seq_length, seq_length, dtype=torch.bool, device=device))
    return mask


def create_padding_mask(
    sequence_lengths: torch.Tensor,
    max_length: int,
    device: torch.device = None
) -> torch.Tensor:
    """
    Create a padding mask for variable-length sequences.
    
    Args:
        sequence_lengths (torch.Tensor): Actual lengths of sequences in batch
        max_length (int): Maximum sequence length
        device (torch.device, optional): Device to place the mask on
    
    Returns:
        torch.Tensor: Boolean mask of shape (batch_size, max_length)
                     True indicates valid (non-padded) positions
    """
    batch_size = sequence_lengths.size(0)
    positions = torch.arange(max_length, device=device or sequence_lengths.device)
    positions = positions.unsqueeze(0).expand(batch_size, -1)
    sequence_lengths = sequence_lengths.unsqueeze(1)
    
    mask = positions < sequence_lengths
    return mask


def compute_attention_scores(
    query: torch.Tensor,
    key: torch.Tensor,
    scale: bool = True
) -> torch.Tensor:
    """
    Compute raw attention scores (before softmax).
    
    This is a utility function that can be used for visualization
    or custom attention implementations.
    
    Args:
        query (torch.Tensor): Query tensor
        key (torch.Tensor): Key tensor
        scale (bool): Whether to apply scaling factor
    
    Returns:
        torch.Tensor: Raw attention scores
    """
    scores = torch.matmul(query, key.transpose(-2, -1))
    
    if scale:
        d_k = query.size(-1)
        scores = scores / math.sqrt(d_k)
    
    return scores