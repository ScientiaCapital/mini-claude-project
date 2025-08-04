"""
Test suite for LoRA (Low-Rank Adaptation) implementation.

This module contains comprehensive tests for the LoRA layer implementation,
ensuring correctness of forward/backward passes, parameter efficiency,
and proper integration with existing neural network architectures.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from typing import Tuple

# Import the LoRA module (to be implemented)
from src.ml.lora import LoRALayer, LoRALinear, calculate_lora_parameters, LoRAConfig


class TestLoRALayer:
    """Test cases for the base LoRA layer implementation."""
    
    def test_lora_layer_initialization(self):
        """Test LoRA layer initialization with various ranks."""
        # Test basic initialization
        in_features, out_features = 768, 768
        rank = 16
        
        lora = LoRALayer(in_features, out_features, rank=rank)
        
        # Check matrix dimensions
        assert lora.lora_A.shape == (rank, in_features)
        assert lora.lora_B.shape == (out_features, rank)
        assert lora.scaling == 1.0  # Default alpha = rank
        
    def test_lora_layer_with_custom_alpha(self):
        """Test LoRA layer with custom alpha scaling."""
        in_features, out_features = 512, 512
        rank = 8
        alpha = 32
        
        lora = LoRALayer(in_features, out_features, rank=rank, alpha=alpha)
        
        # Check scaling factor
        assert lora.scaling == alpha / rank
        
    def test_lora_forward_pass(self):
        """Test forward pass produces correct output shape."""
        in_features, out_features = 768, 768
        rank = 16
        batch_size = 32
        
        lora = LoRALayer(in_features, out_features, rank=rank)
        x = torch.randn(batch_size, in_features)
        
        output = lora(x)
        
        assert output.shape == (batch_size, out_features)
        
    def test_lora_zero_initialization(self):
        """Test that LoRA starts with zero adaptation (no change to base model)."""
        in_features, out_features = 512, 512
        rank = 8
        
        lora = LoRALayer(in_features, out_features, rank=rank)
        x = torch.randn(16, in_features)
        
        # Initially, LoRA should output zeros
        output = lora(x)
        assert torch.allclose(output, torch.zeros_like(output), atol=1e-6)
        
    def test_gradient_flow(self):
        """Test gradient flow through LoRA layers."""
        in_features, out_features = 256, 256
        rank = 4
        
        lora = LoRALayer(in_features, out_features, rank=rank)
        x = torch.randn(8, in_features, requires_grad=True)
        
        output = lora(x)
        loss = output.sum()
        loss.backward()
        
        # Check gradients exist
        assert lora.lora_A.grad is not None
        assert lora.lora_B.grad is not None
        assert x.grad is not None


class TestLoRALinear:
    """Test cases for LoRA-adapted Linear layer."""
    
    def test_lora_linear_initialization(self):
        """Test LoRA Linear layer initialization."""
        in_features, out_features = 768, 768
        rank = 16
        
        layer = LoRALinear(in_features, out_features, rank=rank)
        
        # Check base layer exists
        assert isinstance(layer.base_layer, nn.Linear)
        assert layer.base_layer.in_features == in_features
        assert layer.base_layer.out_features == out_features
        
        # Check LoRA parameters
        assert hasattr(layer, 'lora_A')
        assert hasattr(layer, 'lora_B')
        
    def test_lora_linear_forward(self):
        """Test forward pass of LoRA Linear layer."""
        in_features, out_features = 512, 512
        rank = 8
        batch_size = 16
        
        layer = LoRALinear(in_features, out_features, rank=rank)
        x = torch.randn(batch_size, in_features)
        
        # Test with LoRA disabled
        layer.enable_lora = False
        output_base = layer(x)
        
        # Test with LoRA enabled
        layer.enable_lora = True
        output_lora = layer(x)
        
        assert output_base.shape == output_lora.shape == (batch_size, out_features)
        
    def test_lora_linear_merge_and_unmerge(self):
        """Test merging and unmerging LoRA weights."""
        in_features, out_features = 256, 256
        rank = 4
        
        layer = LoRALinear(in_features, out_features, rank=rank)
        
        # Modify LoRA parameters
        layer.lora_A.data.normal_(0, 0.02)
        layer.lora_B.data.normal_(0, 0.02)
        
        # Get output before merge
        x = torch.randn(8, in_features)
        output_before = layer(x)
        
        # Merge weights
        layer.merge_weights()
        layer.enable_lora = False
        output_merged = layer(x)
        
        # Outputs should be approximately equal
        assert torch.allclose(output_before, output_merged, atol=1e-5)
        
        # Unmerge weights
        layer.unmerge_weights()
        layer.enable_lora = True
        output_unmerged = layer(x)
        
        assert torch.allclose(output_before, output_unmerged, atol=1e-5)


class TestLoRAEfficiency:
    """Test cases for parameter efficiency calculations."""
    
    def test_calculate_lora_parameters(self):
        """Test parameter count calculation for LoRA."""
        # Test case 1: Square matrices
        params = calculate_lora_parameters(768, 768, rank=16)
        expected = 16 * 768 + 768 * 16  # A + B parameters
        assert params == expected
        
        # Test case 2: Non-square matrices
        params = calculate_lora_parameters(512, 1024, rank=8)
        expected = 8 * 512 + 1024 * 8  # A + B parameters
        assert params == expected
        
    def test_parameter_efficiency_ratio(self):
        """Test efficiency ratio calculation."""
        in_features, out_features = 768, 768
        rank = 16
        
        # Calculate base parameters
        base_params = in_features * out_features
        
        # Calculate LoRA parameters
        lora_params = calculate_lora_parameters(in_features, out_features, rank)
        
        # Calculate efficiency ratio
        efficiency_ratio = lora_params / base_params
        
        # Should be much less than 1 for efficiency
        assert efficiency_ratio < 0.1  # Less than 10% of original
        
    def test_memory_efficiency(self):
        """Test memory efficiency of LoRA vs full fine-tuning."""
        in_features, out_features = 1024, 1024
        rank = 8
        batch_size = 32
        
        # Create base layer
        base_layer = nn.Linear(in_features, out_features)
        base_memory = sum(p.numel() * p.element_size() for p in base_layer.parameters())
        
        # Create LoRA layer
        lora_layer = LoRALinear(in_features, out_features, rank=rank)
        lora_memory = sum(
            p.numel() * p.element_size() 
            for name, p in lora_layer.named_parameters() 
            if 'lora' in name
        )
        
        # LoRA should use significantly less memory
        assert lora_memory < base_memory * 0.05  # Less than 5% of base


class TestLoRAConfig:
    """Test cases for LoRA configuration."""
    
    def test_lora_config_defaults(self):
        """Test default LoRA configuration."""
        config = LoRAConfig()
        
        assert config.rank == 16
        assert config.alpha == 16
        assert config.dropout == 0.0
        assert config.target_modules == ['query', 'value']
        
    def test_lora_config_custom(self):
        """Test custom LoRA configuration."""
        config = LoRAConfig(
            rank=32,
            alpha=64,
            dropout=0.1,
            target_modules=['query', 'key', 'value', 'dense']
        )
        
        assert config.rank == 32
        assert config.alpha == 64
        assert config.dropout == 0.1
        assert len(config.target_modules) == 4


class TestLoRAIntegration:
    """Test cases for LoRA integration patterns."""
    
    def test_lora_with_dropout(self):
        """Test LoRA layer with dropout."""
        in_features, out_features = 512, 512
        rank = 8
        dropout = 0.1
        
        config = LoRAConfig(rank=rank, dropout=dropout)
        layer = LoRALinear(in_features, out_features, config=config)
        
        # Initialize LoRA B matrix with small non-zero values so dropout has effect
        with torch.no_grad():
            layer.lora_B.data.normal_(0, 0.01)
        
        # Set to training mode
        layer.train()
        x = torch.randn(100, in_features)
        
        # Run multiple forward passes
        outputs = [layer(x) for _ in range(10)]
        
        # Outputs should be different due to dropout
        for i in range(1, len(outputs)):
            assert not torch.allclose(outputs[0], outputs[i])
            
    def test_lora_gradient_accumulation(self):
        """Test gradient accumulation with LoRA."""
        in_features, out_features = 256, 256
        rank = 4
        
        layer = LoRALinear(in_features, out_features, rank=rank)
        optimizer = torch.optim.AdamW(layer.parameters(), lr=1e-3)
        
        # Accumulate gradients over multiple steps
        accumulation_steps = 4
        for step in range(accumulation_steps):
            x = torch.randn(8, in_features)
            output = layer(x)
            loss = output.sum() / accumulation_steps
            loss.backward()
            
        # Update weights
        optimizer.step()
        optimizer.zero_grad()
        
        # Check that weights were updated
        assert layer.lora_A.grad is None  # Gradients cleared
        assert layer.lora_B.grad is None
        
    def test_lora_mixed_precision(self):
        """Test LoRA with mixed precision training."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
            
        in_features, out_features = 512, 512
        rank = 8
        
        layer = LoRALinear(in_features, out_features, rank=rank).cuda()
        x = torch.randn(16, in_features).cuda()
        
        # Use automatic mixed precision
        with torch.cuda.amp.autocast():
            output = layer(x)
            
        assert output.dtype == torch.float16
        
    def test_rank_selection_impact(self):
        """Test impact of different rank values on parameter efficiency."""
        in_features, out_features = 768, 768
        ranks = [4, 8, 16, 32, 64]
        
        efficiencies = []
        for rank in ranks:
            base_params = in_features * out_features
            lora_params = calculate_lora_parameters(in_features, out_features, rank)
            efficiency = lora_params / base_params
            efficiencies.append(efficiency)
            
        # Efficiency should increase with rank
        for i in range(1, len(efficiencies)):
            assert efficiencies[i] > efficiencies[i-1]
            
        # Even with rank 64, should still be efficient
        assert efficiencies[-1] < 0.25  # Less than 25% of original


if __name__ == "__main__":
    pytest.main([__file__, "-v"])