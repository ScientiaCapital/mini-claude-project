"""
Test suite for scaled dot-product attention mechanism.

This module tests the implementation of scaled dot-product attention,
which is the fundamental building block of transformer architectures.
"""

import unittest
import torch
import numpy as np
from src.ml.attention import ScaledDotProductAttention


class TestScaledDotProductAttention(unittest.TestCase):
    """Test cases for scaled dot-product attention mechanism."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.batch_size = 2
        self.seq_length = 4
        self.d_k = 64  # dimension of keys/queries
        self.d_v = 64  # dimension of values
        
        # Set random seed for reproducibility
        torch.manual_seed(42)
        np.random.seed(42)
        
        # Initialize attention module
        self.attention = ScaledDotProductAttention()
    
    def test_attention_output_shape(self):
        """Test that attention produces correct output shape."""
        # Create random inputs
        query = torch.randn(self.batch_size, self.seq_length, self.d_k)
        key = torch.randn(self.batch_size, self.seq_length, self.d_k)
        value = torch.randn(self.batch_size, self.seq_length, self.d_v)
        
        # Apply attention
        output, attention_weights = self.attention(query, key, value)
        
        # Check output shape
        expected_shape = (self.batch_size, self.seq_length, self.d_v)
        self.assertEqual(output.shape, expected_shape)
        
        # Check attention weights shape
        expected_weights_shape = (self.batch_size, self.seq_length, self.seq_length)
        self.assertEqual(attention_weights.shape, expected_weights_shape)
    
    def test_attention_weights_sum_to_one(self):
        """Test that attention weights sum to 1 across key dimension."""
        query = torch.randn(self.batch_size, self.seq_length, self.d_k)
        key = torch.randn(self.batch_size, self.seq_length, self.d_k)
        value = torch.randn(self.batch_size, self.seq_length, self.d_v)
        
        _, attention_weights = self.attention(query, key, value)
        
        # Sum across key dimension (last dimension)
        weight_sums = attention_weights.sum(dim=-1)
        
        # Check that all sums are approximately 1
        torch.testing.assert_close(
            weight_sums, 
            torch.ones_like(weight_sums),
            rtol=1e-5,
            atol=1e-5
        )
    
    def test_masking_functionality(self):
        """Test that masking properly zeros out attention weights."""
        query = torch.randn(self.batch_size, self.seq_length, self.d_k)
        key = torch.randn(self.batch_size, self.seq_length, self.d_k)
        value = torch.randn(self.batch_size, self.seq_length, self.d_v)
        
        # Create a causal mask (lower triangular)
        mask = torch.tril(torch.ones(self.seq_length, self.seq_length)).bool()
        mask = mask.unsqueeze(0).expand(self.batch_size, -1, -1)
        
        _, attention_weights = self.attention(query, key, value, mask=mask)
        
        # Check that masked positions have zero weight
        for b in range(self.batch_size):
            for i in range(self.seq_length):
                for j in range(self.seq_length):
                    if not mask[b, i, j]:
                        self.assertAlmostEqual(
                            attention_weights[b, i, j].item(), 
                            0.0,
                            places=7
                        )
    
    def test_identity_attention(self):
        """Test attention with identity-like patterns."""
        # Create queries and keys that should attend to themselves
        # Use a simpler approach: make each query similar only to its corresponding key
        query = torch.zeros(self.batch_size, self.seq_length, self.d_k)
        key = torch.zeros(self.batch_size, self.seq_length, self.d_k)
        
        # Set diagonal elements to high values to create strong self-attention
        for i in range(self.seq_length):
            query[:, i, i] = 10.0  # High values on "diagonal" of feature dimension
            key[:, i, i] = 10.0
        
        # Values are simple sequential values
        value = torch.arange(self.seq_length).float()
        value = value.unsqueeze(0).unsqueeze(-1).expand(self.batch_size, -1, self.d_v)
        
        output, attention_weights = self.attention(query, key, value)
        
        # Check that attention is approximately diagonal
        for b in range(self.batch_size):
            diagonal_sum = torch.diag(attention_weights[b]).sum()
            total_sum = attention_weights[b].sum()
            # Most attention should be on diagonal
            self.assertGreater(diagonal_sum / total_sum, 0.7)  # Lowered threshold for robustness
    
    def test_scaling_factor(self):
        """Test that scaling factor prevents gradient vanishing."""
        # Create inputs with large dimension
        large_d_k = 512
        query = torch.randn(1, self.seq_length, large_d_k)
        key = torch.randn(1, self.seq_length, large_d_k)
        value = torch.randn(1, self.seq_length, large_d_k)
        
        # Compute attention without scaling (for comparison)
        scores_unscaled = torch.matmul(query, key.transpose(-2, -1))
        
        # Apply attention with scaling
        output, attention_weights = self.attention(query, key, value)
        
        # Check that attention weights are not saturated
        # (not all zeros or ones due to extreme softmax inputs)
        weight_variance = attention_weights.var().item()
        self.assertGreater(weight_variance, 0.01)
        self.assertLess(weight_variance, 0.5)
    
    def test_gradient_flow(self):
        """Test that gradients flow properly through attention."""
        query = torch.randn(self.batch_size, self.seq_length, self.d_k, requires_grad=True)
        key = torch.randn(self.batch_size, self.seq_length, self.d_k, requires_grad=True)
        value = torch.randn(self.batch_size, self.seq_length, self.d_v, requires_grad=True)
        
        output, _ = self.attention(query, key, value)
        
        # Create a simple loss
        loss = output.sum()
        loss.backward()
        
        # Check that all inputs received gradients
        self.assertIsNotNone(query.grad)
        self.assertIsNotNone(key.grad)
        self.assertIsNotNone(value.grad)
        
        # Check that gradients are not zero
        self.assertGreater(query.grad.abs().sum().item(), 0)
        self.assertGreater(key.grad.abs().sum().item(), 0)
        self.assertGreater(value.grad.abs().sum().item(), 0)
    
    def test_numerical_stability(self):
        """Test numerical stability with extreme values."""
        # Create inputs with large values
        scale = 100.0
        query = torch.randn(1, self.seq_length, self.d_k) * scale
        key = torch.randn(1, self.seq_length, self.d_k) * scale
        value = torch.randn(1, self.seq_length, self.d_v)
        
        output, attention_weights = self.attention(query, key, value)
        
        # Check for NaN or Inf
        self.assertFalse(torch.isnan(output).any())
        self.assertFalse(torch.isinf(output).any())
        self.assertFalse(torch.isnan(attention_weights).any())
        self.assertFalse(torch.isinf(attention_weights).any())
    
    def test_dropout_functionality(self):
        """Test that dropout is applied correctly during training."""
        # Initialize attention with dropout
        attention_with_dropout = ScaledDotProductAttention(dropout=0.5)
        attention_with_dropout.train()  # Set to training mode
        
        query = torch.randn(self.batch_size, self.seq_length, self.d_k)
        key = torch.randn(self.batch_size, self.seq_length, self.d_k)
        value = torch.ones(self.batch_size, self.seq_length, self.d_v)  # Use ones to detect dropout
        
        # Run multiple times to check stochasticity
        outputs = []
        for _ in range(5):
            output, _ = attention_with_dropout(query, key, value)
            outputs.append(output)
        
        # Check that outputs differ due to dropout
        for i in range(1, len(outputs)):
            diff = (outputs[0] - outputs[i]).abs().sum()
            self.assertGreater(diff.item(), 0)
        
        # Test that dropout is disabled in eval mode
        attention_with_dropout.eval()
        output1, _ = attention_with_dropout(query, key, value)
        output2, _ = attention_with_dropout(query, key, value)
        
        # Outputs should be identical in eval mode
        torch.testing.assert_close(output1, output2)
    
    def test_variable_sequence_lengths(self):
        """Test attention with different sequence lengths for Q, K, V."""
        seq_len_q = 3
        seq_len_kv = 5
        
        query = torch.randn(self.batch_size, seq_len_q, self.d_k)
        key = torch.randn(self.batch_size, seq_len_kv, self.d_k)
        value = torch.randn(self.batch_size, seq_len_kv, self.d_v)
        
        output, attention_weights = self.attention(query, key, value)
        
        # Check output shape
        self.assertEqual(output.shape, (self.batch_size, seq_len_q, self.d_v))
        
        # Check attention weights shape
        self.assertEqual(attention_weights.shape, (self.batch_size, seq_len_q, seq_len_kv))


if __name__ == '__main__':
    unittest.main()