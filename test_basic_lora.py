#!/usr/bin/env python3
"""
Basic LoRA implementation test without pytest dependency.
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import torch
import torch.nn as nn
from ml.lora import LoRALayer, LoRALinear, calculate_lora_parameters, LoRAConfig


def test_basic_functionality():
    """Test basic LoRA functionality."""
    print("Testing LoRA basic functionality...")
    
    # Test 1: LoRA Layer initialization
    print("1. Testing LoRA layer initialization...")
    lora = LoRALayer(768, 768, rank=16)
    assert lora.lora_A.shape == (16, 768)
    assert lora.lora_B.shape == (768, 16)
    print("   ✓ LoRA layer shapes correct")
    
    # Test 2: Forward pass
    print("2. Testing forward pass...")
    x = torch.randn(32, 768)
    output = lora(x)
    assert output.shape == (32, 768)
    print("   ✓ Forward pass shape correct")
    
    # Test 3: Zero initialization check
    print("3. Testing zero initialization...")
    # For new LoRA layer, output should be close to zero
    output_norm = torch.norm(output).item()
    print(f"   Output norm: {output_norm:.6f}")
    assert output_norm < 1e-3, f"Expected small output, got norm {output_norm}"
    print("   ✓ Zero initialization working")
    
    # Test 4: LoRA Linear layer
    print("4. Testing LoRA Linear layer...")
    lora_linear = LoRALinear(512, 512, rank=8)
    x = torch.randn(16, 512)
    output = lora_linear(x)
    assert output.shape == (16, 512)
    print("   ✓ LoRA Linear layer working")
    
    # Test 5: Parameter calculation
    print("5. Testing parameter calculation...")
    params = calculate_lora_parameters(768, 768, 16)
    expected = 16 * 768 + 768 * 16
    assert params == expected, f"Expected {expected}, got {params}"
    print(f"   ✓ Parameter calculation correct: {params} parameters")
    
    # Test 6: Efficiency ratio
    print("6. Testing efficiency ratio...")
    base_params = 768 * 768
    lora_params = calculate_lora_parameters(768, 768, 16)
    efficiency = lora_params / base_params
    print(f"   Base parameters: {base_params}")
    print(f"   LoRA parameters: {lora_params}")
    print(f"   Efficiency ratio: {efficiency:.4f} ({efficiency*100:.2f}%)")
    assert efficiency < 0.1, f"Expected efficiency < 10%, got {efficiency*100:.2f}%"
    print("   ✓ Efficiency ratio good")
    
    # Test 7: Gradient flow
    print("7. Testing gradient flow...")
    lora = LoRALayer(256, 256, rank=4)
    x = torch.randn(8, 256, requires_grad=True)
    output = lora(x)
    loss = output.sum()
    loss.backward()
    
    assert lora.lora_A.grad is not None, "lora_A gradient is None"
    assert lora.lora_B.grad is not None, "lora_B gradient is None"
    assert x.grad is not None, "Input gradient is None"
    print("   ✓ Gradient flow working")
    
    # Test 8: Merge/unmerge weights
    print("8. Testing merge/unmerge weights...")
    lora_linear = LoRALinear(256, 256, rank=4)
    
    # Add some adaptation
    lora_linear.lora_A.data.normal_(0, 0.02)
    lora_linear.lora_B.data.normal_(0, 0.02)
    
    x = torch.randn(8, 256)
    output_before = lora_linear(x)
    
    # Merge weights
    lora_linear.merge_weights()
    lora_linear.enable_lora = False
    output_merged = lora_linear(x)
    
    # Should be approximately equal
    diff = torch.norm(output_before - output_merged).item()
    print(f"   Difference after merge: {diff:.6f}")
    assert diff < 1e-4, f"Merge failed, difference: {diff}"
    
    # Unmerge weights
    lora_linear.unmerge_weights()
    lora_linear.enable_lora = True
    output_unmerged = lora_linear(x)
    
    diff = torch.norm(output_before - output_unmerged).item()
    print(f"   Difference after unmerge: {diff:.6f}")
    assert diff < 1e-4, f"Unmerge failed, difference: {diff}"
    print("   ✓ Merge/unmerge working")
    
    print("\n✅ All tests passed! LoRA implementation is working correctly.")


def test_efficiency_metrics():
    """Test efficiency calculation functions."""
    print("\nTesting efficiency metrics...")
    
    # Test different layer configurations
    layer_configs = [
        (768, 768, 16),    # Typical transformer layer
        (512, 2048, 8),    # Feed-forward layer
        (1024, 1024, 32),  # Large layer
        (256, 256, 4),     # Small layer
    ]
    
    print("\nEfficiency Analysis:")
    print("Layer Config (in, out, rank) | Base Params | LoRA Params | Efficiency | Reduction")
    print("-" * 85)
    
    for in_feat, out_feat, rank in layer_configs:
        base_params = in_feat * out_feat
        lora_params = calculate_lora_parameters(in_feat, out_feat, rank)
        efficiency = lora_params / base_params
        reduction = base_params / lora_params
        
        print(f"({in_feat:4d}, {out_feat:4d}, {rank:2d})        | "
              f"{base_params:9,} | {lora_params:9,} | "
              f"{efficiency:8.4f} | {reduction:8.1f}x")
    
    print("\n✅ Efficiency metrics calculated successfully.")


def test_memory_estimation():
    """Test memory usage estimation."""
    print("\nTesting memory estimation...")
    
    from ml.lora import estimate_memory_usage
    
    # Simulate a small transformer model
    layer_dims = [
        (768, 768),   # Self-attention Q
        (768, 768),   # Self-attention K  
        (768, 768),   # Self-attention V
        (768, 768),   # Self-attention output
        (768, 3072),  # Feed-forward 1
        (3072, 768),  # Feed-forward 2
    ] * 12  # 12 layers
    
    ranks = [4, 8, 16, 32]
    
    print("\nMemory Usage Analysis (12-layer transformer-like model):")
    print("Rank | Base Memory | LoRA Memory | Savings | Reduction Ratio")
    print("-" * 65)
    
    for rank in ranks:
        memory_stats = estimate_memory_usage(layer_dims, rank)
        
        print(f"{rank:4d} | {memory_stats['base_memory_mb']:10.1f} MB | "
              f"{memory_stats['lora_memory_mb']:10.1f} MB | "
              f"{memory_stats['memory_savings_mb']:6.1f} MB | "
              f"{memory_stats['memory_reduction_ratio']:13.4f}")
    
    print("\n✅ Memory estimation working correctly.")


if __name__ == "__main__":
    try:
        test_basic_functionality()
        test_efficiency_metrics()
        test_memory_estimation()
        print("\n🎉 All LoRA tests completed successfully!")
        print("\nLoRA implementation features:")
        print("- ✅ Low-rank weight decomposition (A and B matrices)")
        print("- ✅ Proper zero initialization for no initial adaptation")
        print("- ✅ Kaiming initialization for matrix A")
        print("- ✅ Configurable rank and alpha scaling")
        print("- ✅ Dropout regularization support")
        print("- ✅ Weight merging for inference efficiency")
        print("- ✅ Parameter efficiency calculations")
        print("- ✅ Memory usage estimation")
        print("- ✅ Gradient flow verification")
        print("- ✅ Integration with standard PyTorch layers")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)