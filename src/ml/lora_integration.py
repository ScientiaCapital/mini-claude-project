"""
LoRA Integration Examples

This module provides practical examples of how to integrate LoRA
with existing neural network models for parameter-efficient fine-tuning.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any
from .lora import LoRALinear, LoRAConfig, LoRAManager


class TransformerAttention(nn.Module):
    """Example transformer attention layer for LoRA integration."""
    
    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.output = nn.Linear(hidden_size, hidden_size)
        
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        
        q = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Simplified attention (without proper scaling and masking)
        attn_weights = torch.matmul(q, k.transpose(-2, -1))
        attn_weights = torch.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_weights, v)
        
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.hidden_size
        )
        
        output = self.output(attn_output)
        return output


class TransformerLayer(nn.Module):
    """Example transformer layer with attention and feed-forward."""
    
    def __init__(self, hidden_size: int, num_heads: int, intermediate_size: int):
        super().__init__()
        self.attention = TransformerAttention(hidden_size, num_heads)
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_size, intermediate_size),
            nn.ReLU(),
            nn.Linear(intermediate_size, hidden_size)
        )
        self.layer_norm1 = nn.LayerNorm(hidden_size)
        self.layer_norm2 = nn.LayerNorm(hidden_size)
        
    def forward(self, x):
        # Attention with residual connection
        attn_output = self.attention(x)
        x = self.layer_norm1(x + attn_output)
        
        # Feed-forward with residual connection
        ff_output = self.feed_forward(x)
        x = self.layer_norm2(x + ff_output)
        
        return x


class ExampleTransformerModel(nn.Module):
    """Example transformer model for LoRA demonstration."""
    
    def __init__(
        self,
        vocab_size: int = 30000,
        hidden_size: int = 768,
        num_layers: int = 12,
        num_heads: int = 12,
        intermediate_size: int = 3072,
        max_position_embeddings: int = 512
    ):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.position_embedding = nn.Embedding(max_position_embeddings, hidden_size)
        
        self.layers = nn.ModuleList([
            TransformerLayer(hidden_size, num_heads, intermediate_size)
            for _ in range(num_layers)
        ])
        
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.output_projection = nn.Linear(hidden_size, vocab_size)
        
    def forward(self, input_ids: torch.Tensor, position_ids: Optional[torch.Tensor] = None):
        batch_size, seq_len = input_ids.shape
        
        if position_ids is None:
            position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        
        # Embeddings
        token_embeddings = self.embedding(input_ids)
        position_embeddings = self.position_embedding(position_ids)
        embeddings = token_embeddings + position_embeddings
        
        # Transformer layers
        hidden_states = embeddings
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        
        hidden_states = self.layer_norm(hidden_states)
        
        # Output projection
        logits = self.output_projection(hidden_states)
        
        return logits


def apply_lora_to_model(
    model: nn.Module,
    config: LoRAConfig,
    target_modules: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Apply LoRA adaptation to a model.
    
    Args:
        model: PyTorch model to adapt
        config: LoRA configuration
        target_modules: List of module names to target
        
    Returns:
        Dictionary with application statistics
    """
    if target_modules is None:
        target_modules = ['query', 'key', 'value', 'output', 'feed_forward.0', 'feed_forward.2']
    
    manager = LoRAManager(model, config)
    stats = manager.apply_lora(target_modules)
    
    return {
        'manager': manager,
        'stats': stats,
        'efficiency_report': manager.get_efficiency_report()
    }


def create_lora_training_setup(
    model: nn.Module,
    lora_config: LoRAConfig,
    learning_rate: float = 1e-4,
    weight_decay: float = 0.01
) -> Dict[str, Any]:
    """
    Set up a model for LoRA training.
    
    Args:
        model: Base model to adapt
        lora_config: LoRA configuration
        learning_rate: Learning rate for LoRA parameters
        weight_decay: Weight decay for regularization
        
    Returns:
        Dictionary containing model, optimizer, and metadata
    """
    # Apply LoRA to model
    lora_result = apply_lora_to_model(model, lora_config)
    manager = lora_result['manager']
    
    # Freeze base model parameters
    for name, param in model.named_parameters():
        if 'lora' not in name:
            param.requires_grad = False
    
    # Create optimizer for LoRA parameters only
    lora_parameters = [p for name, p in model.named_parameters() if 'lora' in name and p.requires_grad]
    
    optimizer = torch.optim.AdamW(
        lora_parameters,
        lr=learning_rate,
        weight_decay=weight_decay
    )
    
    return {
        'model': model,
        'optimizer': optimizer,
        'manager': manager,
        'lora_parameters': lora_parameters,
        'stats': lora_result['stats'],
        'efficiency_report': lora_result['efficiency_report']
    }


def demonstrate_lora_fine_tuning():
    """
    Demonstrate LoRA fine-tuning workflow.
    
    This function shows the complete workflow for applying LoRA
    to a model and setting up training.
    """
    print("LoRA Fine-tuning Demonstration")
    print("=" * 50)
    
    # Create a small model for demonstration
    model = ExampleTransformerModel(
        vocab_size=1000,
        hidden_size=256,
        num_layers=4,
        num_heads=8,
        intermediate_size=1024
    )
    
    print(f"Original model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Configure LoRA
    lora_config = LoRAConfig(
        rank=16,
        alpha=32,
        dropout=0.1,
        target_modules=['query', 'key', 'value', 'output']
    )
    
    # Set up LoRA training
    training_setup = create_lora_training_setup(model, lora_config)
    
    print(f"\nLoRA Configuration:")
    print(f"- Rank: {lora_config.rank}")
    print(f"- Alpha: {lora_config.alpha}")
    print(f"- Dropout: {lora_config.dropout}")
    print(f"- Target modules: {lora_config.target_modules}")
    
    print(f"\nTraining Setup:")
    print(f"- Trainable LoRA parameters: {len(training_setup['lora_parameters'])}")
    print(f"- Total LoRA parameters: {sum(p.numel() for p in training_setup['lora_parameters']):,}")
    
    efficiency_report = training_setup['efficiency_report']
    print(f"\nEfficiency Report:")
    print(f"- Base parameters: {efficiency_report['total_base_parameters']:,}")
    print(f"- LoRA parameters: {efficiency_report['total_lora_parameters']:,}")
    print(f"- Efficiency ratio: {efficiency_report['overall_efficiency_ratio']:.4f}")
    print(f"- Memory savings: {efficiency_report['memory_savings_pct']:.1f}%")
    
    # Simulate training step
    print(f"\nSimulating training step...")
    model.train()
    
    # Create dummy input
    batch_size, seq_len = 4, 32
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    target_ids = torch.randint(0, 1000, (batch_size, seq_len))
    
    # Forward pass
    logits = model(input_ids)
    loss = nn.CrossEntropyLoss()(logits.view(-1, logits.size(-1)), target_ids.view(-1))
    
    # Backward pass
    training_setup['optimizer'].zero_grad()
    loss.backward()
    training_setup['optimizer'].step()
    
    print(f"- Loss: {loss.item():.4f}")
    print(f"- Gradients computed for LoRA parameters only")
    
    # Demonstrate weight merging for inference
    print(f"\nDemonstrating inference optimization...")
    model.eval()
    
    with torch.no_grad():
        # Before merging
        start_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
        end_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
        
        if start_time:
            start_time.record()
        
        logits_before = model(input_ids)
        
        if end_time:
            end_time.record()
            torch.cuda.synchronize()
            time_before = start_time.elapsed_time(end_time)
        else:
            time_before = "N/A (CPU)"
        
        # Merge weights
        training_setup['manager'].merge_all_weights()
        
        if start_time:
            start_time.record()
        
        logits_after = model(input_ids)
        
        if end_time:
            end_time.record()
            torch.cuda.synchronize()
            time_after = start_time.elapsed_time(end_time)
        else:
            time_after = "N/A (CPU)"
        
        # Check equivalence
        max_diff = torch.max(torch.abs(logits_before - logits_after)).item()
        print(f"- Maximum difference after merging: {max_diff:.2e}")
        print(f"- Inference time before merge: {time_before}")
        print(f"- Inference time after merge: {time_after}")
        
        # Unmerge for continued training
        training_setup['manager'].unmerge_all_weights()
    
    print(f"\n✅ LoRA fine-tuning demonstration complete!")
    
    return training_setup


def compare_training_modes():
    """Compare full fine-tuning vs LoRA fine-tuning."""
    print("\n" + "=" * 50)
    print("Training Mode Comparison")
    print("=" * 50)
    
    # Create model
    model = ExampleTransformerModel(
        vocab_size=1000,
        hidden_size=512,
        num_layers=6,
        num_heads=8,
        intermediate_size=2048
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params_full = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Base Model:")
    print(f"- Total parameters: {total_params:,}")
    print(f"- Trainable (full fine-tuning): {trainable_params_full:,}")
    
    # Configure LoRA
    lora_config = LoRAConfig(rank=16, alpha=32)
    lora_setup = create_lora_training_setup(model, lora_config)
    
    trainable_params_lora = sum(p.numel() for p in lora_setup['lora_parameters'])
    
    print(f"\nLoRA Fine-tuning:")
    print(f"- Trainable LoRA parameters: {trainable_params_lora:,}")
    print(f"- Parameter reduction: {trainable_params_full / trainable_params_lora:.1f}x")
    print(f"- Training efficiency: {trainable_params_lora / trainable_params_full:.4f}")
    
    # Memory estimation (simplified)
    memory_full = trainable_params_full * 4 * 3  # params + gradients + optimizer states
    memory_lora = trainable_params_lora * 4 * 3
    
    print(f"\nMemory Usage (estimated):")
    print(f"- Full fine-tuning: {memory_full / (1024**2):.1f} MB")
    print(f"- LoRA fine-tuning: {memory_lora / (1024**2):.1f} MB")
    print(f"- Memory savings: {(memory_full - memory_lora) / (1024**2):.1f} MB")
    print(f"- Memory reduction: {memory_full / memory_lora:.1f}x")


if __name__ == "__main__":
    try:
        # Run demonstrations
        training_setup = demonstrate_lora_fine_tuning()
        compare_training_modes()
        
        print(f"\n🎉 LoRA integration examples completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error in demonstration: {e}")
        import traceback
        traceback.print_exc()