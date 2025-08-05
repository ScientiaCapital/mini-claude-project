"""
LoRA (Low-Rank Adaptation) Implementation

This module implements LoRA and QLoRA layers for parameter-efficient fine-tuning
of large neural networks. LoRA decomposes weight updates into low-rank matrices,
significantly reducing the number of trainable parameters while maintaining
model performance.

Mathematical Foundation:
For a weight matrix W ∈ R^(d×k), LoRA represents the update as:
W' = W + BA
where B ∈ R^(d×r) and A ∈ R^(r×k) with rank r << min(d,k)

The forward pass becomes:
h = W₀x + BAx = W₀x + B(Ax)

Memory Efficiency:
- Original parameters: d × k
- LoRA parameters: r × (d + k)
- Reduction ratio: r(d+k)/(dk) = r/k + r/d

References:
- Hu et al. (2021): "LoRA: Low-Rank Adaptation of Large Language Models"
- Dettmers et al. (2023): "QLoRA: Efficient Finetuning of Quantized LLMs"
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import List, Optional, Union, Dict, Any


@dataclass
class LoRAConfig:
    """Configuration class for LoRA parameters."""
    
    rank: int = 16
    alpha: int = 16
    dropout: float = 0.0
    target_modules: List[str] = None
    bias: str = "none"  # "none", "all", or "lora_only"
    
    def __post_init__(self):
        if self.target_modules is None:
            self.target_modules = ['query', 'value']


class LoRALayer(nn.Module):
    """
    Base LoRA layer implementation.
    
    This class implements the core LoRA functionality with low-rank decomposition
    of weight updates. The layer computes the adaptation as h = BAx where
    B and A are low-rank matrices.
    
    Args:
        in_features (int): Number of input features
        out_features (int): Number of output features  
        rank (int): Rank of the decomposition (r)
        alpha (float): Scaling factor, typically set to rank
        dropout (float): Dropout probability for regularization
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 16,
        alpha: float = None,
        dropout: float = 0.0
    ):
        super().__init__()
        
        if rank > min(in_features, out_features):
            raise ValueError(
                f"Rank {rank} must be <= min(in_features={in_features}, "
                f"out_features={out_features})={min(in_features, out_features)}"
            )
            
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha if alpha is not None else rank
        self.scaling = self.alpha / rank
        
        # Initialize low-rank matrices
        # A: (rank, in_features) - initialized with Kaiming normal
        # B: (out_features, rank) - initialized to zero for zero-init adaptation
        self.lora_A = nn.Parameter(torch.empty(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        
        self.reset_parameters()
        
    def reset_parameters(self):
        """Initialize LoRA parameters using proper initialization strategies."""
        # Initialize A with Kaiming normal (similar to original paper)
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        # B is already initialized to zero in __init__
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of LoRA layer.
        
        Computes: scaling * B @ (A @ x.T).T = scaling * B @ (dropout(A @ x.T)).T
        
        Args:
            x: Input tensor of shape (..., in_features)
            
        Returns:
            Output tensor of shape (..., out_features)
        """
        # x: (..., in_features)
        # A: (rank, in_features)
        # B: (out_features, rank)
        
        # Compute A @ x.T -> (rank, ...)
        ax = F.linear(x, self.lora_A)  # (..., rank)
        
        # Apply dropout for regularization
        ax = self.dropout(ax)
        
        # Compute B @ (A @ x) -> (..., out_features)
        result = F.linear(ax, self.lora_B) * self.scaling
        
        return result


class LoRALinear(nn.Module):
    """
    LoRA-adapted Linear layer.
    
    This class wraps a standard Linear layer with LoRA adaptation,
    allowing for parameter-efficient fine-tuning while preserving
    the original model weights.
    
    The forward pass computes: W₀x + αBAx where:
    - W₀ is the frozen base weight matrix
    - B, A are the low-rank adaptation matrices
    - α is the scaling factor
    
    Args:
        in_features (int): Input feature dimension
        out_features (int): Output feature dimension
        rank (int): LoRA rank
        config (LoRAConfig, optional): LoRA configuration
        bias (bool): Whether to use bias in base layer
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 16,
        config: Optional[LoRAConfig] = None,
        bias: bool = True
    ):
        super().__init__()
        
        if config is None:
            config = LoRAConfig(rank=rank)
        
        self.in_features = in_features
        self.out_features = out_features
        self.rank = config.rank
        self.config = config
        
        # Base linear layer (frozen during training)
        self.base_layer = nn.Linear(in_features, out_features, bias=bias)
        
        # LoRA adaptation layers
        self.lora_A = nn.Parameter(torch.empty(config.rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, config.rank))
        
        # Scaling and dropout
        self.scaling = config.alpha / config.rank
        self.dropout = nn.Dropout(config.dropout) if config.dropout > 0.0 else nn.Identity()
        
        # Control flags
        self.enable_lora = True
        self.merged = False
        
        self.reset_parameters()
        
    def reset_parameters(self):
        """Initialize LoRA parameters."""
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        # lora_B is already zero-initialized
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with optional LoRA adaptation.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor with base transformation and optional LoRA adaptation
        """
        # Base transformation
        result = self.base_layer(x)
        
        # Add LoRA adaptation if enabled and not merged
        if self.enable_lora and not self.merged:
            # Compute LoRA adaptation: α * B @ (dropout(A @ x))
            lora_result = F.linear(x, self.lora_A)  # (..., rank)
            lora_result = self.dropout(lora_result)
            lora_result = F.linear(lora_result, self.lora_B)  # (..., out_features)
            result += lora_result * self.scaling
            
        return result
        
    def merge_weights(self):
        """
        Merge LoRA weights into the base layer for inference efficiency.
        
        This operation modifies the base layer weights to include the LoRA
        adaptation, allowing for faster inference without separate LoRA computation.
        """
        if not self.merged:
            # Compute LoRA weight update: α * B @ A
            lora_weight = self.lora_B @ self.lora_A * self.scaling
            self.base_layer.weight.data += lora_weight
            self.merged = True
            
    def unmerge_weights(self):
        """
        Unmerge LoRA weights from the base layer.
        
        This reverses the merge operation, restoring the original base weights
        and enabling separate LoRA computation.
        """
        if self.merged:
            # Remove LoRA weight update: subtract α * B @ A
            lora_weight = self.lora_B @ self.lora_A * self.scaling
            self.base_layer.weight.data -= lora_weight
            self.merged = False


def calculate_lora_parameters(
    in_features: int,
    out_features: int,
    rank: int
) -> int:
    """
    Calculate the number of parameters in a LoRA adaptation.
    
    For a weight matrix W ∈ R^(out_features × in_features), LoRA adds:
    - A matrix: rank × in_features parameters
    - B matrix: out_features × rank parameters
    
    Args:
        in_features: Input dimension
        out_features: Output dimension  
        rank: LoRA rank
        
    Returns:
        Total number of LoRA parameters
    """
    return rank * in_features + out_features * rank


def calculate_parameter_efficiency(
    in_features: int,
    out_features: int,
    rank: int
) -> Dict[str, Union[int, float]]:
    """
    Calculate parameter efficiency metrics for LoRA adaptation.
    
    Args:
        in_features: Input dimension
        out_features: Output dimension
        rank: LoRA rank
        
    Returns:
        Dictionary containing efficiency metrics
    """
    base_params = in_features * out_features
    lora_params = calculate_lora_parameters(in_features, out_features, rank)
    
    efficiency_ratio = lora_params / base_params
    reduction_factor = base_params / lora_params
    memory_savings = 1 - efficiency_ratio
    
    return {
        'base_parameters': base_params,
        'lora_parameters': lora_params,
        'efficiency_ratio': efficiency_ratio,
        'reduction_factor': reduction_factor,
        'memory_savings_pct': memory_savings * 100,
        'rank': rank
    }


def estimate_memory_usage(
    layer_dims: List[tuple],
    rank: int,
    dtype: torch.dtype = torch.float32
) -> Dict[str, float]:
    """
    Estimate memory usage for LoRA adaptation across multiple layers.
    
    Args:
        layer_dims: List of (in_features, out_features) tuples
        rank: LoRA rank
        dtype: Parameter data type
        
    Returns:
        Memory usage estimates in MB
    """
    bytes_per_param = torch.tensor(0, dtype=dtype).element_size()
    
    total_base_params = sum(in_feat * out_feat for in_feat, out_feat in layer_dims)
    total_lora_params = sum(
        calculate_lora_parameters(in_feat, out_feat, rank)
        for in_feat, out_feat in layer_dims
    )
    
    base_memory_mb = (total_base_params * bytes_per_param) / (1024 * 1024)
    lora_memory_mb = (total_lora_params * bytes_per_param) / (1024 * 1024)
    
    return {
        'base_memory_mb': base_memory_mb,
        'lora_memory_mb': lora_memory_mb,
        'memory_savings_mb': base_memory_mb - lora_memory_mb,
        'memory_reduction_ratio': lora_memory_mb / base_memory_mb
    }


class LoRAManager:
    """
    Utility class for managing LoRA adaptations across a model.
    
    This class provides convenience methods for applying LoRA to specific
    modules, managing training/inference modes, and computing efficiency metrics.
    """
    
    def __init__(self, model: nn.Module, config: LoRAConfig):
        self.model = model
        self.config = config
        self.lora_layers = {}
        
    def apply_lora(self, target_modules: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Apply LoRA adaptation to specified modules.
        
        Args:
            target_modules: List of module names to adapt
            
        Returns:
            Dictionary with application statistics
        """
        if target_modules is None:
            target_modules = self.config.target_modules
            
        adapted_count = 0
        total_params_added = 0
        
        for name, module in self.model.named_modules():
            if any(target in name for target in target_modules):
                if isinstance(module, nn.Linear):
                    # Replace Linear with LoRALinear
                    lora_layer = LoRALinear(
                        module.in_features,
                        module.out_features,
                        config=self.config,
                        bias=module.bias is not None
                    )
                    
                    # Copy original weights
                    lora_layer.base_layer.weight.data = module.weight.data.clone()
                    if module.bias is not None:
                        lora_layer.base_layer.bias.data = module.bias.data.clone()
                    
                    # Replace module
                    self._replace_module(name, lora_layer)
                    self.lora_layers[name] = lora_layer
                    
                    adapted_count += 1
                    total_params_added += calculate_lora_parameters(
                        module.in_features, module.out_features, self.config.rank
                    )
        
        return {
            'adapted_layers': adapted_count,
            'parameters_added': total_params_added,
            'target_modules': target_modules
        }
    
    def _replace_module(self, module_name: str, new_module: nn.Module):
        """Replace a module in the model by name."""
        parent = self.model
        components = module_name.split('.')
        
        for component in components[:-1]:
            parent = getattr(parent, component)
            
        setattr(parent, components[-1], new_module)
    
    def merge_all_weights(self):
        """Merge all LoRA weights for inference."""
        for layer in self.lora_layers.values():
            layer.merge_weights()
    
    def unmerge_all_weights(self):
        """Unmerge all LoRA weights for training."""
        for layer in self.lora_layers.values():
            layer.unmerge_weights()
    
    def get_efficiency_report(self) -> Dict[str, Any]:
        """Generate comprehensive efficiency report."""
        total_base_params = 0
        total_lora_params = 0
        layer_details = []
        
        for name, layer in self.lora_layers.items():
            base_params = layer.in_features * layer.out_features
            lora_params = calculate_lora_parameters(
                layer.in_features, layer.out_features, layer.rank
            )
            
            total_base_params += base_params
            total_lora_params += lora_params
            
            layer_details.append({
                'name': name,
                'base_params': base_params,
                'lora_params': lora_params,
                'efficiency_ratio': lora_params / base_params
            })
        
        overall_efficiency = total_lora_params / total_base_params if total_base_params > 0 else 0
        
        return {
            'total_base_parameters': total_base_params,
            'total_lora_parameters': total_lora_params,
            'overall_efficiency_ratio': overall_efficiency,
            'memory_savings_pct': (1 - overall_efficiency) * 100,
            'layer_details': layer_details,
            'config': {
                'rank': self.config.rank,
                'alpha': self.config.alpha,
                'dropout': self.config.dropout
            }
        }


# Utility functions for rank selection and optimization
def suggest_optimal_rank(
    in_features: int,
    out_features: int,
    target_efficiency: float = 0.1,
    max_rank: Optional[int] = None
) -> int:
    """
    Suggest optimal LoRA rank based on efficiency target.
    
    Args:
        in_features: Input dimension
        out_features: Output dimension
        target_efficiency: Target parameter efficiency ratio (0-1)
        max_rank: Maximum allowed rank
        
    Returns:
        Suggested rank value
    """
    base_params = in_features * out_features
    
    if max_rank is None:
        max_rank = min(in_features, out_features) // 2
    
    for rank in range(1, max_rank + 1):
        lora_params = calculate_lora_parameters(in_features, out_features, rank)
        efficiency = lora_params / base_params
        
        if efficiency >= target_efficiency:
            return max(1, rank - 1)  # Return previous rank that was under target
    
    return max_rank


def benchmark_rank_efficiency(
    layer_dims: List[tuple],
    ranks: List[int],
    dtype: torch.dtype = torch.float32
) -> Dict[int, Dict[str, float]]:
    """
    Benchmark efficiency across different rank values.
    
    Args:
        layer_dims: List of (in_features, out_features) tuples
        ranks: List of rank values to test
        dtype: Parameter data type
        
    Returns:
        Dictionary mapping rank to efficiency metrics
    """
    results = {}
    
    for rank in ranks:
        memory_stats = estimate_memory_usage(layer_dims, rank, dtype)
        
        total_base_params = sum(in_feat * out_feat for in_feat, out_feat in layer_dims)
        total_lora_params = sum(
            calculate_lora_parameters(in_feat, out_feat, rank)
            for in_feat, out_feat in layer_dims
        )
        
        results[rank] = {
            'total_parameters': total_lora_params,
            'efficiency_ratio': total_lora_params / total_base_params,
            'memory_mb': memory_stats['lora_memory_mb'],
            'memory_reduction': memory_stats['memory_reduction_ratio']
        }
    
    return results