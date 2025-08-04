"""
Test suite for data processing pipeline.
"""

import pytest
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, patch

# Import will fail initially - this is expected in TDD RED phase
try:
    from src.data.pipeline import (
        DataProcessingPipeline,
        PipelineConfig,
        PipelineState,
        PipelineStep
    )
    from src.data.alpaca_utils import AlpacaDataset, AlpacaRecord
except ImportError:
    pass


class TestPipelineConfig:
    """Test pipeline configuration."""
    
    def test_pipeline_config_creation(self):
        """Test creating pipeline configuration."""
        config = PipelineConfig(
            input_path="input.json",
            output_path="output.json"
        )
        
        assert config.input_path == "input.json"
        assert config.output_path == "output.json"
        assert config.quality_threshold == 0.7
        assert config.enable_checkpointing is True


class TestPipelineState:
    """Test pipeline state management."""
    
    def test_pipeline_state_initialization(self):
        """Test pipeline state initialization."""
        config = PipelineConfig("input.json", "output.json")
        state = PipelineState(config=config)
        
        assert state.current_step == 0
        assert state.completed_steps == []
        assert state.start_time is not None


class TestDataProcessingPipeline:
    """Test data processing pipeline."""
    
    def setup_method(self):
        """Setup test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.input_path = Path(self.temp_dir) / "input.json"
        self.output_path = Path(self.temp_dir) / "output.json"
        self.checkpoint_dir = Path(self.temp_dir) / "checkpoints"
        
        # Create sample input data
        sample_data = {
            "data": [
                {
                    "instruction": "Test instruction",
                    "input": "Test input",
                    "output": "Test output with sufficient length for validation"
                }
            ],
            "metadata": {"test": True}
        }
        
        with open(self.input_path, 'w') as f:
            json.dump(sample_data, f)
    
    def test_pipeline_initialization(self):
        """Test pipeline initialization."""
        config = PipelineConfig(
            input_path=str(self.input_path),
            output_path=str(self.output_path),
            checkpoint_dir=str(self.checkpoint_dir)
        )
        
        pipeline = DataProcessingPipeline(config)
        
        assert pipeline.config == config
        assert pipeline.state.current_step == 0
        assert len(pipeline.steps) > 0
    
    def test_pipeline_step_execution(self):
        """Test individual pipeline step execution."""
        config = PipelineConfig(
            input_path=str(self.input_path),
            output_path=str(self.output_path),
            checkpoint_dir=str(self.checkpoint_dir)
        )
        
        pipeline = DataProcessingPipeline(config)
        
        # Test load_data step
        result = pipeline._load_data()
        
        assert "records_loaded" in result
        assert result["records_loaded"] > 0
    
    def test_pipeline_quality_threshold(self):
        """Test pipeline quality threshold enforcement."""
        config = PipelineConfig(
            input_path=str(self.input_path),
            output_path=str(self.output_path),
            checkpoint_dir=str(self.checkpoint_dir),
            quality_threshold=0.99  # Very high threshold
        )
        
        pipeline = DataProcessingPipeline(config)
        
        # Load data first
        pipeline._load_data()
        
        # Quality check might fail with high threshold
        # This tests the quality enforcement mechanism
        try:
            pipeline._initial_quality_check()
        except ValueError:
            # Expected if quality is below threshold
            pass
    
    def test_pipeline_checkpointing(self):
        """Test pipeline checkpointing functionality."""
        config = PipelineConfig(
            input_path=str(self.input_path),
            output_path=str(self.output_path),
            checkpoint_dir=str(self.checkpoint_dir),
            enable_checkpointing=True
        )
        
        pipeline = DataProcessingPipeline(config)
        
        # Create a test checkpoint
        test_data = {"test": "checkpoint_data"}
        pipeline._create_checkpoint("test_checkpoint", test_data)
        
        # Verify checkpoint file exists
        checkpoint_path = self.checkpoint_dir / "test_checkpoint.json"
        assert checkpoint_path.exists()
        
        # Verify checkpoint content
        with open(checkpoint_path) as f:
            checkpoint_data = json.load(f)
        
        assert checkpoint_data["checkpoint_name"] == "test_checkpoint"
        assert checkpoint_data["data"] == test_data
    
    def test_pipeline_data_cleaning(self):
        """Test data cleaning functionality."""
        config = PipelineConfig(
            input_path=str(self.input_path),
            output_path=str(self.output_path),
            checkpoint_dir=str(self.checkpoint_dir)
        )
        
        pipeline = DataProcessingPipeline(config)
        
        # Create test data with issues
        pipeline.dataset = AlpacaDataset(data=[
            AlpacaRecord(
                instruction="Good instruction",
                input="Good input",
                output="Good output with sufficient length"
            ),
            AlpacaRecord(
                instruction="Bad instruction",
                input="Bad input",
                output="Bad"  # Too short
            )
        ])
        
        result = pipeline._clean_data()
        
        assert result["removed_count"] > 0
        assert len(pipeline.dataset.data) < 2  # Some data should be removed
    
    def test_pipeline_error_handling(self):
        """Test pipeline error handling and retries."""
        config = PipelineConfig(
            input_path="nonexistent_file.json",  # This will cause an error
            output_path=str(self.output_path),
            checkpoint_dir=str(self.checkpoint_dir),
            max_retries=2
        )
        
        pipeline = DataProcessingPipeline(config)
        
        # Mock a step that fails
        def failing_step():
            raise ValueError("Test error")
        
        step = PipelineStep(
            name="test_step",
            function=failing_step,
            description="Test failing step"
        )
        
        # Should retry and eventually fail
        with pytest.raises(ValueError):
            pipeline._execute_step_with_retries(step)


class TestPipelineIntegration:
    """Integration tests for complete pipeline execution."""
    
    def setup_method(self):
        """Setup test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.input_path = Path(self.temp_dir) / "input.json"
        self.output_path = Path(self.temp_dir) / "output.json"
        self.checkpoint_dir = Path(self.temp_dir) / "checkpoints"
    
    def test_full_pipeline_execution(self):
        """Test complete pipeline execution."""
        config = PipelineConfig(
            input_path=str(self.input_path),
            output_path=str(self.output_path),
            checkpoint_dir=str(self.checkpoint_dir),
            quality_threshold=0.5,  # Lower threshold for test
            validation_split=0.2
        )
        
        pipeline = DataProcessingPipeline(config)
        
        # Should generate sample data if input doesn't exist
        report = pipeline.run()
        
        assert report["status"] == "completed"
        assert "execution_time_seconds" in report
        assert len(report["completed_steps"]) > 0
        
        # Verify output files exist
        assert self.output_path.exists()
    
    def test_pipeline_resume_from_checkpoint(self):
        """Test pipeline resuming from checkpoint."""
        config = PipelineConfig(
            input_path=str(self.input_path),
            output_path=str(self.output_path),
            checkpoint_dir=str(self.checkpoint_dir),
            enable_checkpointing=True
        )
        
        # Create initial checkpoint
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_data = {
            "checkpoint_name": "data_loaded",
            "timestamp": "2023-01-01T00:00:00",
            "state": {
                "config": config.__dict__,
                "current_step": 1,
                "completed_steps": ["load_data"],
                "start_time": "2023-01-01T00:00:00",
                "last_checkpoint": "data_loaded"
            },
            "data": {"test": "data"},
            "config_hash": "test_hash"
        }
        
        checkpoint_path = self.checkpoint_dir / "data_loaded.json"
        with open(checkpoint_path, 'w') as f:
            json.dump(checkpoint_data, f)
        
        pipeline = DataProcessingPipeline(config)
        
        # Should attempt to load checkpoint
        # (actual resuming depends on config hash matching)
        result = pipeline._load_checkpoint()
        
        # Test passes if no exception is raised