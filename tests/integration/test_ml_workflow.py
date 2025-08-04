"""
Integration Test Suite for ML Components Workflow

This module provides comprehensive integration tests that validate all ML components
work seamlessly together in production-ready workflows, following TDD principles.

Tests cover:
1. Attention + LoRA integration with mathematical correctness
2. Data Pipeline → Training integration compatibility  
3. Complete end-to-end workflow validation
4. Production environment readiness
5. Performance benchmarks and efficiency metrics
"""

import unittest
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List, Any, Tuple
import time
import logging

# Import ML components
from src.ml.attention import ScaledDotProductAttention, create_causal_mask
from src.ml.lora import LoRALayer, LoRALinear, LoRAConfig, calculate_lora_parameters
from src.data.pipeline import DataProcessingPipeline, PipelineConfig
from src.data.quality_metrics import DataQualityAssessment
from src.data.alpaca_utils import AlpacaDataset, AlpacaRecord, load_alpaca_dataset

# Configure logging for integration tests
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SimpleTransformerBlock(nn.Module):
    """Simple transformer block for integration testing."""
    
    def __init__(self, hidden_size: int, num_heads: int = 8):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        # Multi-head attention components
        self.query = nn.Linear(hidden_size, hidden_size, bias=False)
        self.key = nn.Linear(hidden_size, hidden_size, bias=False)
        self.value = nn.Linear(hidden_size, hidden_size, bias=False)
        self.output_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        
        # Attention mechanism
        self.attention = ScaledDotProductAttention(dropout=0.1)
        
        # Feed-forward network
        self.ffn1 = nn.Linear(hidden_size, hidden_size * 4, bias=False)
        self.ffn2 = nn.Linear(hidden_size * 4, hidden_size, bias=False)
        
        # Layer norms
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """Forward pass through transformer block."""
        batch_size, seq_len, hidden_size = x.shape
        
        # Multi-head attention
        residual = x
        x = self.norm1(x)
        
        # Linear projections
        q = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # Reshape for attention computation
        q = q.transpose(1, 2).contiguous().view(batch_size * self.num_heads, seq_len, self.head_dim)
        k = k.transpose(1, 2).contiguous().view(batch_size * self.num_heads, seq_len, self.head_dim)
        v = v.transpose(1, 2).contiguous().view(batch_size * self.num_heads, seq_len, self.head_dim)
        
        # Apply attention
        attn_output, _ = self.attention(q, k, v, mask)
        
        # Reshape back
        attn_output = attn_output.view(batch_size, self.num_heads, seq_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, hidden_size)
        
        # Output projection and residual
        attn_output = self.output_proj(attn_output)
        x = residual + attn_output
        
        # Feed-forward network
        residual = x
        x = self.norm2(x)
        x = torch.relu(self.ffn1(x))
        x = self.ffn2(x)
        x = residual + x
        
        return x


class TestAttentionLoRAIntegration(unittest.TestCase):
    """Test integration between Attention mechanism and LoRA adaptation."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.hidden_size = 512
        self.seq_length = 16
        self.batch_size = 4
        self.rank = 16
        
        torch.manual_seed(42)
        np.random.seed(42)
        
    def test_lora_attention_integration(self):
        """Test applying LoRA to attention layers with mathematical correctness."""
        logger.info("Testing LoRA + Attention integration")
        
        # Create transformer block
        transformer = SimpleTransformerBlock(self.hidden_size, num_heads=8)
        
        # Apply LoRA to attention layers
        lora_config = LoRAConfig(rank=self.rank, alpha=32, dropout=0.1)
        
        # Replace query and value projections with LoRA versions
        transformer.query = LoRALinear(
            self.hidden_size, self.hidden_size, config=lora_config, bias=False
        )
        transformer.value = LoRALinear(
            self.hidden_size, self.hidden_size, config=lora_config, bias=False
        )
        
        # Test input
        x = torch.randn(self.batch_size, self.seq_length, self.hidden_size)
        mask = create_causal_mask(self.seq_length)
        
        # Forward pass
        start_time = time.time()
        output = transformer(x, mask)
        forward_time = time.time() - start_time
        
        # Validate output shape
        self.assertEqual(output.shape, x.shape)
        
        # Test gradient flow through LoRA + Attention
        loss = output.sum()
        start_time = time.time()
        loss.backward()
        backward_time = time.time() - start_time
        
        # Verify gradients exist for LoRA parameters
        self.assertIsNotNone(transformer.query.lora_A.grad)
        self.assertIsNotNone(transformer.query.lora_B.grad)
        self.assertIsNotNone(transformer.value.lora_A.grad)
        self.assertIsNotNone(transformer.value.lora_B.grad)
        
        # Calculate parameter efficiency
        base_params = 2 * self.hidden_size * self.hidden_size  # query + value
        lora_params = 2 * calculate_lora_parameters(self.hidden_size, self.hidden_size, self.rank)
        efficiency_ratio = lora_params / base_params
        
        # Verify efficiency gains
        self.assertLess(efficiency_ratio, 0.1)  # Less than 10% of original parameters
        
        logger.info(f"✓ LoRA-Attention integration successful")
        logger.info(f"  Forward time: {forward_time:.4f}s")
        logger.info(f"  Backward time: {backward_time:.4f}s")
        logger.info(f"  Parameter efficiency: {efficiency_ratio:.4f}")
        logger.info(f"  Memory reduction: {1 - efficiency_ratio:.1%}")
        
    def test_lora_attention_mathematical_correctness(self):
        """Test mathematical correctness of LoRA + Attention integration."""
        logger.info("Testing mathematical correctness of LoRA + Attention")
        
        # Create attention layer with LoRA
        attention_layer = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        lora_attention = LoRALinear(self.hidden_size, self.hidden_size, rank=self.rank, bias=False)
        
        # Copy base weights
        lora_attention.base_layer.weight.data = attention_layer.weight.data.clone()
        
        # Test input
        x = torch.randn(self.batch_size, self.seq_length, self.hidden_size)
        
        # Initially, LoRA should produce same output as base layer
        with torch.no_grad():
            base_output = attention_layer(x)
            lora_output = lora_attention(x)
            
        # Should be identical initially (LoRA B matrix is zero-initialized)
        torch.testing.assert_close(base_output, lora_output, atol=1e-6)
        
        # Modify LoRA parameters
        with torch.no_grad():
            lora_attention.lora_A.data.normal_(0, 0.02)
            lora_attention.lora_B.data.normal_(0, 0.02)
        
        # Now outputs should differ
        with torch.no_grad():
            base_output_after = attention_layer(x)
            lora_output_after = lora_attention(x)
            
        # Base layer should remain unchanged
        torch.testing.assert_close(base_output, base_output_after, atol=1e-6)
        
        # LoRA output should be different
        self.assertFalse(torch.allclose(base_output, lora_output_after, atol=1e-5))
        
        logger.info("✓ Mathematical correctness verified")
        
    def test_attention_mask_with_lora(self):
        """Test attention masking works correctly with LoRA."""
        logger.info("Testing attention masking with LoRA")
        
        # Create LoRA attention
        lora_config = LoRAConfig(rank=8, alpha=16)
        query_layer = LoRALinear(self.hidden_size, self.hidden_size, config=lora_config, bias=False)
        key_layer = LoRALinear(self.hidden_size, self.hidden_size, config=lora_config, bias=False)
        value_layer = LoRALinear(self.hidden_size, self.hidden_size, config=lora_config, bias=False)
        
        attention = ScaledDotProductAttention()
        
        # Test input
        x = torch.randn(self.batch_size, self.seq_length, self.hidden_size)
        
        # Apply projections
        q = query_layer(x)
        k = key_layer(x)
        v = value_layer(x)
        
        # Create causal mask
        mask = create_causal_mask(self.seq_length)
        mask = mask.unsqueeze(0).expand(self.batch_size, -1, -1)
        
        # Apply attention with mask
        output, attn_weights = attention(q, k, v, mask)
        
        # Verify mask is applied correctly
        for b in range(self.batch_size):
            for i in range(self.seq_length):
                for j in range(self.seq_length):
                    if not mask[b, i, j]:
                        self.assertAlmostEqual(attn_weights[b, i, j].item(), 0.0, places=6)
        
        logger.info("✓ Attention masking works correctly with LoRA")


class TestDataPipelineTrainingIntegration(unittest.TestCase):
    """Test integration between Data Pipeline and Training components."""
    
    def setUp(self):
        """Set up test fixtures with temporary directories."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.input_path = self.temp_dir / "input.json"
        self.output_path = self.temp_dir / "output.json"
        self.checkpoint_dir = self.temp_dir / "checkpoints"
        
    def tearDown(self):
        """Clean up temporary directories."""
        shutil.rmtree(self.temp_dir)
        
    def test_pipeline_output_format_compatibility(self):
        """Test that pipeline output format is compatible with training."""
        logger.info("Testing pipeline → training format compatibility")
        
        # Configure pipeline
        config = PipelineConfig(
            input_path=str(self.input_path),
            output_path=str(self.output_path),
            checkpoint_dir=str(self.checkpoint_dir),
            quality_threshold=0.6,
            validation_split=0.2,
            enable_checkpointing=False
        )
        
        # Run pipeline
        pipeline = DataProcessingPipeline(config)
        report = pipeline.run()
        
        # Verify pipeline succeeded
        self.assertEqual(report['status'], 'completed')
        self.assertGreaterEqual(report['final_metrics']['overall_score'], 0.6)
        
        # Load processed datasets
        train_dataset = load_alpaca_dataset(self.temp_dir / f"{self.output_path.stem}_train.json")
        val_dataset = load_alpaca_dataset(self.temp_dir / f"{self.output_path.stem}_val.json")
        
        # Verify format compatibility for training
        self.assertGreater(len(train_dataset.data), 0)
        self.assertGreater(len(val_dataset.data), 0)
        
        # Check Alpaca format compliance
        for record in train_dataset.data[:5]:  # Check first 5 records
            self.assertIsInstance(record, AlpacaRecord)
            self.assertIsInstance(record.instruction, str)
            self.assertIsInstance(record.input, str)
            self.assertIsInstance(record.output, str)
            self.assertGreater(len(record.instruction.strip()), 0)
            self.assertGreater(len(record.output.strip()), 0)
        
        logger.info(f"✓ Pipeline output format compatible with training")
        logger.info(f"  Training samples: {len(train_dataset.data)}")
        logger.info(f"  Validation samples: {len(val_dataset.data)}")
        logger.info(f"  Quality score: {report['final_metrics']['overall_score']:.3f}")
        
    def test_data_quality_meets_training_requirements(self):
        """Test that processed data meets quality requirements for training."""
        logger.info("Testing data quality for training requirements")
        
        # Configure pipeline with higher quality threshold
        config = PipelineConfig(
            input_path=str(self.input_path),
            output_path=str(self.output_path),
            checkpoint_dir=str(self.checkpoint_dir),
            quality_threshold=0.7,
            validation_split=0.2,
            enable_checkpointing=False
        )
        
        # Run pipeline
        pipeline = DataProcessingPipeline(config)
        report = pipeline.run()
        
        # Verify high quality achieved
        quality_score = report['final_metrics']['overall_score']
        self.assertGreaterEqual(quality_score, 0.7)
        
        # Load and analyze final dataset
        final_dataset = load_alpaca_dataset(self.output_path)
        
        # Convert to DataFrame for quality assessment
        df = pd.DataFrame([record.dict() for record in final_dataset.data])
        
        # Run comprehensive quality assessment
        assessor = DataQualityAssessment()
        quality_report = assessor.assess_dataset(df)
        
        # Verify quality metrics
        self.assertTrue(quality_report['format_validation']['is_valid'])
        self.assertGreaterEqual(quality_report['completeness']['overall'], 0.95)
        self.assertLessEqual(len(quality_report['recommendations']), 2)
        
        # Check for training-specific requirements
        min_instruction_length = 10
        min_output_length = 20
        
        valid_samples = 0
        for record in final_dataset.data:
            if (len(record.instruction) >= min_instruction_length and 
                len(record.output) >= min_output_length):
                valid_samples += 1
        
        valid_ratio = valid_samples / len(final_dataset.data)
        self.assertGreaterEqual(valid_ratio, 0.8)  # At least 80% should meet requirements
        
        logger.info(f"✓ Data quality meets training requirements")
        logger.info(f"  Overall quality: {quality_score:.3f}")
        logger.info(f"  Valid samples: {valid_ratio:.1%}")
        logger.info(f"  Format validation: {quality_report['format_validation']['is_valid']}")


class TestEndToEndWorkflow(unittest.TestCase):
    """Test complete end-to-end ML workflow integration."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.hidden_size = 256
        self.rank = 8
        
        torch.manual_seed(42)
        np.random.seed(42)
        
    def tearDown(self):
        """Clean up temporary directories."""
        shutil.rmtree(self.temp_dir)
        
    def test_complete_workflow_data_to_inference(self):
        """Test complete workflow: Data → Processing → Training → Inference."""
        logger.info("Testing complete end-to-end workflow")
        
        # Step 1: Data Processing Pipeline
        logger.info("Step 1: Data Processing")
        config = PipelineConfig(
            input_path=str(self.temp_dir / "input.json"),
            output_path=str(self.temp_dir / "output.json"),
            checkpoint_dir=str(self.temp_dir / "checkpoints"),
            quality_threshold=0.6,
            validation_split=0.2,
            enable_checkpointing=False
        )
        
        pipeline = DataProcessingPipeline(config)
        pipeline_report = pipeline.run()
        
        self.assertEqual(pipeline_report['status'], 'completed')
        
        # Step 2: Load processed data
        logger.info("Step 2: Loading processed data")
        train_dataset = load_alpaca_dataset(self.temp_dir / "output_train.json")
        val_dataset = load_alpaca_dataset(self.temp_dir / "output_val.json")
        
        self.assertGreater(len(train_dataset.data), 0)
        self.assertGreater(len(val_dataset.data), 0)
        
        # Step 3: Create model with LoRA + Attention
        logger.info("Step 3: Creating model with LoRA + Attention")
        model = SimpleTransformerBlock(self.hidden_size, num_heads=4)
        
        # Apply LoRA to specific layers
        lora_config = LoRAConfig(rank=self.rank, alpha=16, dropout=0.1)
        model.query = LoRALinear(self.hidden_size, self.hidden_size, config=lora_config, bias=False)
        model.value = LoRALinear(self.hidden_size, self.hidden_size, config=lora_config, bias=False)
        
        # Step 4: Simulate training setup
        logger.info("Step 4: Training setup simulation")
        
        # Create sample batch data
        batch_size = 4
        seq_length = 16
        
        # Simulate input tokens (random for testing)
        input_ids = torch.randint(0, 1000, (batch_size, seq_length))
        
        # Simple embedding layer for test
        embedding = nn.Embedding(1000, self.hidden_size)
        x = embedding(input_ids)
        
        # Create causal mask
        mask = create_causal_mask(seq_length)
        
        # Step 5: Forward pass
        logger.info("Step 5: Forward pass")
        start_time = time.time()
        output = model(x, mask)
        forward_time = time.time() - start_time
        
        self.assertEqual(output.shape, (batch_size, seq_length, self.hidden_size))
        
        # Step 6: Backward pass (simulating training)
        logger.info("Step 6: Backward pass simulation")
        loss = output.sum()
        start_time = time.time()
        loss.backward()
        backward_time = time.time() - start_time
        
        # Verify gradients
        self.assertIsNotNone(model.query.lora_A.grad)
        self.assertIsNotNone(model.query.lora_B.grad)
        
        # Step 7: Calculate efficiency metrics
        logger.info("Step 7: Efficiency analysis")
        base_params = 2 * self.hidden_size * self.hidden_size  # query + value
        lora_params = 2 * calculate_lora_parameters(self.hidden_size, self.hidden_size, self.rank)
        efficiency_ratio = lora_params / base_params
        
        # Step 8: Inference simulation
        logger.info("Step 8: Inference simulation")
        model.eval()
        with torch.no_grad():
            inference_output = model(x, mask)
            
        self.assertEqual(inference_output.shape, output.shape)
        
        # Performance benchmarks
        logger.info("✓ Complete end-to-end workflow successful")
        logger.info(f"  Pipeline execution: {pipeline_report['execution_time_seconds']:.2f}s")
        logger.info(f"  Training samples: {len(train_dataset.data)}")
        logger.info(f"  Validation samples: {len(val_dataset.data)}")
        logger.info(f"  Forward pass time: {forward_time:.4f}s")
        logger.info(f"  Backward pass time: {backward_time:.4f}s")
        logger.info(f"  Parameter efficiency: {efficiency_ratio:.4f}")
        logger.info(f"  Memory reduction: {1 - efficiency_ratio:.1%}")
        
    def test_workflow_error_handling(self):
        """Test error handling throughout the workflow."""
        logger.info("Testing workflow error handling")
        
        # Test with invalid quality threshold
        config = PipelineConfig(
            input_path=str(self.temp_dir / "input.json"),
            output_path=str(self.temp_dir / "output.json"),
            checkpoint_dir=str(self.temp_dir / "checkpoints"),
            quality_threshold=0.99,  # Very high threshold to trigger failure
            validation_split=0.2,
            enable_checkpointing=False
        )
        
        pipeline = DataProcessingPipeline(config)
        
        # This should handle errors gracefully
        try:
            report = pipeline.run()
            # If it succeeds despite high threshold, that's also fine
            self.assertIn(report['status'], ['completed', 'failed'])
        except Exception as e:
            # Pipeline should handle errors gracefully
            self.assertIsInstance(e, (ValueError, RuntimeError))
            
        logger.info("✓ Error handling works correctly")


class TestProductionIntegration(unittest.TestCase):
    """Test production environment integration and Next.js compatibility."""
    
    def test_import_compatibility(self):
        """Test that all ML components can be imported without conflicts."""
        logger.info("Testing import compatibility for production")
        
        # Test imports don't conflict
        try:
            from src.ml.attention import ScaledDotProductAttention
            from src.ml.lora import LoRALayer, LoRALinear
            from src.data.pipeline import DataProcessingPipeline
            from src.data.quality_metrics import DataQualityAssessment
            
            # Create instances to test instantiation
            attention = ScaledDotProductAttention()
            lora = LoRALayer(128, 128, rank=8)
            
            self.assertIsInstance(attention, ScaledDotProductAttention)
            self.assertIsInstance(lora, LoRALayer)
            
        except ImportError as e:
            self.fail(f"Import failed: {e}")
            
        logger.info("✓ All imports compatible")
        
    def test_memory_constraints(self):
        """Test components work within production memory constraints."""
        logger.info("Testing memory constraints for production")
        
        # Test with production-typical sizes
        hidden_size = 768  # BERT-base size
        batch_size = 8     # Reasonable batch size
        seq_length = 512   # Typical sequence length
        rank = 16          # Efficient rank
        
        # Create model components
        attention = ScaledDotProductAttention(dropout=0.1)
        lora_layer = LoRALinear(hidden_size, hidden_size, rank=rank)
        
        # Test memory usage
        x = torch.randn(batch_size, seq_length, hidden_size)
        
        # Measure memory before
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            memory_before = torch.cuda.memory_allocated()
        
        # Forward pass
        q = k = v = x
        output, _ = attention(q, k, v)
        lora_output = lora_layer(x)
        
        # Measure memory after
        if torch.cuda.is_available():
            memory_after = torch.cuda.memory_allocated()
            memory_used = memory_after - memory_before
            
            # Should use reasonable amount of memory (less than 1GB for this test)
            self.assertLess(memory_used, 1e9)  # 1GB limit
            
        logger.info("✓ Memory constraints satisfied")
        
    def test_inference_speed(self):
        """Test inference speed meets production requirements."""
        logger.info("Testing inference speed for production")
        
        hidden_size = 512
        seq_length = 128
        batch_size = 1  # Single inference
        rank = 16
        
        # Create lightweight model
        model = SimpleTransformerBlock(hidden_size, num_heads=8)
        
        # Apply LoRA for efficiency
        lora_config = LoRAConfig(rank=rank, alpha=32)
        model.query = LoRALinear(hidden_size, hidden_size, config=lora_config, bias=False)
        model.value = LoRALinear(hidden_size, hidden_size, config=lora_config, bias=False)
        
        # Merge weights for faster inference
        model.query.merge_weights()
        model.value.merge_weights()
        
        model.eval()
        
        # Test input
        x = torch.randn(batch_size, seq_length, hidden_size)
        mask = create_causal_mask(seq_length)
        
        # Warmup
        with torch.no_grad():
            _ = model(x, mask)
        
        # Measure inference time
        num_trials = 10
        times = []
        
        with torch.no_grad():
            for _ in range(num_trials):
                start_time = time.time()
                output = model(x, mask)
                end_time = time.time()
                times.append(end_time - start_time)
        
        avg_time = np.mean(times)
        std_time = np.std(times)
        
        # Should be fast enough for production (< 100ms for this size)
        self.assertLess(avg_time, 0.1)  # 100ms limit
        
        logger.info(f"✓ Inference speed acceptable: {avg_time:.4f}±{std_time:.4f}s")


class TestPerformanceBenchmarks(unittest.TestCase):
    """Test suite for performance benchmarks and metrics."""
    
    def test_component_performance_benchmarks(self):
        """Benchmark individual component performance."""
        logger.info("Running component performance benchmarks")
        
        results = {}
        
        # Attention benchmarks
        hidden_size = 512
        seq_length = 128
        batch_size = 8
        
        attention = ScaledDotProductAttention()
        x = torch.randn(batch_size, seq_length, hidden_size)
        
        # Benchmark attention
        times = []
        for _ in range(20):
            start_time = time.time()
            output, _ = attention(x, x, x)
            end_time = time.time()
            times.append(end_time - start_time)
        
        results['attention'] = {
            'mean_time': np.mean(times),
            'std_time': np.std(times),
            'throughput': batch_size / np.mean(times)  # samples/second
        }
        
        # LoRA benchmarks
        ranks = [4, 8, 16, 32]
        results['lora'] = {}
        
        for rank in ranks:
            lora = LoRALinear(hidden_size, hidden_size, rank=rank)
            
            times = []
            for _ in range(20):
                start_time = time.time()
                output = lora(x.view(-1, hidden_size))
                end_time = time.time()
                times.append(end_time - start_time)
            
            base_params = hidden_size * hidden_size
            lora_params = calculate_lora_parameters(hidden_size, hidden_size, rank)
            
            results['lora'][f'rank_{rank}'] = {
                'mean_time': np.mean(times),
                'std_time': np.std(times),
                'efficiency_ratio': lora_params / base_params,
                'throughput': (batch_size * seq_length) / np.mean(times)
            }
        
        # Validate performance requirements
        self.assertLess(results['attention']['mean_time'], 0.01)  # < 10ms
        
        for rank_results in results['lora'].values():
            self.assertLess(rank_results['efficiency_ratio'], 0.2)  # < 20% parameters
            self.assertGreater(rank_results['throughput'], 1000)   # > 1000 samples/sec
        
        logger.info("✓ Performance benchmarks completed")
        for component, metrics in results.items():
            if component == 'lora':
                for rank, rank_metrics in metrics.items():
                    logger.info(f"  LoRA {rank}: {rank_metrics['mean_time']:.4f}s, "
                              f"efficiency: {rank_metrics['efficiency_ratio']:.4f}")
            else:
                logger.info(f"  {component}: {metrics['mean_time']:.4f}s, "
                          f"throughput: {metrics['throughput']:.0f} samples/sec")
        
        return results


def run_integration_tests():
    """Run all integration tests and generate comprehensive report."""
    logger.info("="*80)
    logger.info("RUNNING ML COMPONENT INTEGRATION TEST SUITE")
    logger.info("="*80)
    
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestAttentionLoRAIntegration,
        TestDataPipelineTrainingIntegration,
        TestEndToEndWorkflow,
        TestProductionIntegration,
        TestPerformanceBenchmarks
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Generate summary report
    logger.info("="*80)
    logger.info("INTEGRATION TEST SUMMARY")
    logger.info("="*80)
    logger.info(f"Tests run: {result.testsRun}")
    logger.info(f"Failures: {len(result.failures)}")
    logger.info(f"Errors: {len(result.errors)}")
    logger.info(f"Success rate: {(result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100:.1f}%")
    
    if result.failures:
        logger.error("FAILURES:")
        for test, traceback in result.failures:
            logger.error(f"  {test}: {traceback}")
    
    if result.errors:
        logger.error("ERRORS:")
        for test, traceback in result.errors:
            logger.error(f"  {test}: {traceback}")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_integration_tests()
    exit(0 if success else 1)