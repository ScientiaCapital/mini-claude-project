"""
Test suite for experiment tracking functionality.
Following TDD principles - these tests define the expected behavior.
"""

import pytest
import tempfile
import shutil
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
from unittest.mock import Mock, patch, MagicMock


class TestExperimentTracker:
    """Test cases for the main ExperimentTracker class."""
    
    def setup_method(self):
        """Set up test environment before each test."""
        self.temp_dir = tempfile.mkdtemp()
        self.tracking_uri = f"file://{self.temp_dir}"
        
    def teardown_method(self):
        """Clean up after each test."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_experiment_tracker_initialization(self):
        """Test that ExperimentTracker initializes correctly."""
        from src.mlops.experiment_tracking import ExperimentTracker
        
        tracker = ExperimentTracker(
            tracking_uri=self.tracking_uri,
            experiment_name="test_experiment"
        )
        
        assert tracker.tracking_uri == self.tracking_uri
        assert tracker.experiment_name == "test_experiment"
        assert tracker.experiment_id is not None
        assert tracker.run_id is None  # No active run initially
    
    def test_start_run_creates_new_run(self):
        """Test that start_run creates a new MLflow run."""
        from src.mlops.experiment_tracking import ExperimentTracker
        
        tracker = ExperimentTracker(
            tracking_uri=self.tracking_uri,
            experiment_name="test_experiment"
        )
        
        run_name = "test_run_001"
        tracker.start_run(run_name=run_name)
        
        assert tracker.run_id is not None
        assert tracker.active_run is not None
        assert tracker.active_run.info.run_name == run_name
    
    def test_start_run_with_tags(self):
        """Test that start_run properly sets tags."""
        from src.mlops.experiment_tracking import ExperimentTracker
        
        tracker = ExperimentTracker(
            tracking_uri=self.tracking_uri,
            experiment_name="test_experiment"
        )
        
        tags = {
            "model_type": "transformer",
            "task": "text_classification",
            "version": "1.0.0"
        }
        
        tracker.start_run(run_name="test_run", tags=tags)
        
        # Verify tags are set
        for key, value in tags.items():
            assert tracker.get_tag(key) == value
    
    def test_log_params(self):
        """Test parameter logging functionality."""
        from src.mlops.experiment_tracking import ExperimentTracker
        
        tracker = ExperimentTracker(
            tracking_uri=self.tracking_uri,
            experiment_name="test_experiment"
        )
        
        tracker.start_run(run_name="test_run")
        
        params = {
            "learning_rate": 0.001,
            "batch_size": 32,
            "epochs": 10,
            "model_name": "bert-base-uncased"
        }
        
        tracker.log_params(params)
        
        # Verify params are logged
        for key, value in params.items():
            assert tracker.get_param(key) == str(value)
    
    def test_log_metrics(self):
        """Test metric logging functionality."""
        from src.mlops.experiment_tracking import ExperimentTracker
        
        tracker = ExperimentTracker(
            tracking_uri=self.tracking_uri,
            experiment_name="test_experiment"
        )
        
        tracker.start_run(run_name="test_run")
        
        # Log metrics at different steps
        tracker.log_metric("accuracy", 0.85, step=1)
        tracker.log_metric("loss", 0.23, step=1)
        tracker.log_metric("accuracy", 0.87, step=2)
        tracker.log_metric("loss", 0.21, step=2)
        
        # Verify metrics are logged
        assert tracker.get_metric("accuracy") == 0.87  # Latest value
        assert tracker.get_metric("loss") == 0.21
    
    def test_log_artifacts(self):
        """Test artifact logging functionality."""
        from src.mlops.experiment_tracking import ExperimentTracker
        
        tracker = ExperimentTracker(
            tracking_uri=self.tracking_uri,
            experiment_name="test_experiment"
        )
        
        tracker.start_run(run_name="test_run")
        
        # Create test artifacts
        test_file = Path(self.temp_dir) / "test_model.pt"
        test_file.write_text("fake model content")
        
        config_file = Path(self.temp_dir) / "config.json"
        config_file.write_text('{"param": "value"}')
        
        # Log artifacts
        tracker.log_artifact(str(test_file), "models")
        tracker.log_artifact(str(config_file), "configs")
        
        # Verify artifacts are logged
        artifacts = tracker.list_artifacts()
        artifact_paths = [artifact.path for artifact in artifacts]
        
        assert "models/test_model.pt" in artifact_paths
        assert "configs/config.json" in artifact_paths
    
    def test_log_model(self):
        """Test model logging with metadata."""
        from src.mlops.experiment_tracking import ExperimentTracker
        
        tracker = ExperimentTracker(
            tracking_uri=self.tracking_uri,
            experiment_name="test_experiment"
        )
        
        tracker.start_run(run_name="test_run")
        
        # Mock model
        mock_model = Mock()
        
        model_info = {
            "architecture": "transformer",
            "num_parameters": 110000000,
            "task": "text_classification"
        }
        
        tracker.log_model(
            model=mock_model,
            artifact_path="model",
            model_info=model_info
        )
        
        # Verify model is logged with metadata
        assert tracker.get_tag("mlflow.model_artifact_path") == "model"
        for key, value in model_info.items():
            assert tracker.get_tag(f"model.{key}") == str(value)
    
    def test_end_run(self):
        """Test that end_run properly closes the active run."""
        from src.mlops.experiment_tracking import ExperimentTracker
        
        tracker = ExperimentTracker(
            tracking_uri=self.tracking_uri,
            experiment_name="test_experiment"
        )
        
        tracker.start_run(run_name="test_run")
        assert tracker.active_run is not None
        
        tracker.end_run()
        assert tracker.active_run is None
        assert tracker.run_id is None
    
    def test_context_manager(self):
        """Test that ExperimentTracker works as a context manager."""
        from src.mlops.experiment_tracking import ExperimentTracker
        
        tracker = ExperimentTracker(
            tracking_uri=self.tracking_uri,
            experiment_name="test_experiment"
        )
        
        with tracker.run("test_run") as run_tracker:
            assert run_tracker.active_run is not None
            run_tracker.log_param("test_param", "test_value")
        
        # Run should be ended after context exit
        assert tracker.active_run is None


class TestExperimentTemplate:
    """Test cases for experiment templates."""
    
    def test_fine_tuning_template_creation(self):
        """Test creation of fine-tuning experiment template."""
        from src.mlops.experiment_tracking import FineTuningTemplate
        
        template = FineTuningTemplate(
            model_name="bert-base-uncased",
            dataset_name="glue_cola",
            task_type="classification"
        )
        
        assert template.model_name == "bert-base-uncased"
        assert template.dataset_name == "glue_cola"
        assert template.task_type == "classification"
        assert template.experiment_name == "fine_tuning_bert-base-uncased_glue_cola"
    
    def test_fine_tuning_template_params(self):
        """Test that fine-tuning template generates proper default parameters."""
        from src.mlops.experiment_tracking import FineTuningTemplate
        
        template = FineTuningTemplate(
            model_name="bert-base-uncased",
            dataset_name="glue_cola",
            task_type="classification"
        )
        
        params = template.get_default_params()
        
        expected_params = {
            "model_name", "dataset_name", "task_type", "learning_rate",
            "batch_size", "epochs", "warmup_steps", "weight_decay",
            "max_length", "optimizer"
        }
        
        assert set(params.keys()) == expected_params
        assert params["model_name"] == "bert-base-uncased"
        assert params["dataset_name"] == "glue_cola"
        assert params["task_type"] == "classification"
    
    def test_pretraining_template_creation(self):
        """Test creation of pretraining experiment template."""
        from src.mlops.experiment_tracking import PretrainingTemplate
        
        template = PretrainingTemplate(
            model_architecture="gpt2",
            dataset_name="openwebtext",
            model_size="small"
        )
        
        assert template.model_architecture == "gpt2"
        assert template.dataset_name == "openwebtext"
        assert template.model_size == "small"
        assert template.experiment_name == "pretraining_gpt2_small_openwebtext"
    
    def test_evaluation_template_creation(self):
        """Test creation of evaluation experiment template."""
        from src.mlops.experiment_tracking import EvaluationTemplate
        
        template = EvaluationTemplate(
            model_name="fine_tuned_bert",
            benchmark_name="glue",
            model_version="1.0.0"
        )
        
        assert template.model_name == "fine_tuned_bert"
        assert template.benchmark_name == "glue"
        assert template.model_version == "1.0.0"
        assert template.experiment_name == "evaluation_fine_tuned_bert_glue_v1.0.0"


class TestMetricsLogger:
    """Test cases for advanced metrics logging functionality."""
    
    def test_metrics_logger_initialization(self):
        """Test MetricsLogger initialization."""
        from src.mlops.experiment_tracking import MetricsLogger
        
        logger = MetricsLogger()
        
        assert logger.metrics_buffer == {}
        assert logger.step_counter == 0
    
    def test_log_training_metrics(self):
        """Test logging of training metrics."""
        from src.mlops.experiment_tracking import MetricsLogger
        
        logger = MetricsLogger()
        
        metrics = {
            "train_loss": 0.45,
            "train_accuracy": 0.87,
            "learning_rate": 0.001
        }
        
        logger.log_training_metrics(metrics, step=1)
        
        assert logger.metrics_buffer["train_loss"] == [(1, 0.45)]
        assert logger.metrics_buffer["train_accuracy"] == [(1, 0.87)]
        assert logger.metrics_buffer["learning_rate"] == [(1, 0.001)]
    
    def test_log_validation_metrics(self):
        """Test logging of validation metrics."""
        from src.mlops.experiment_tracking import MetricsLogger
        
        logger = MetricsLogger()
        
        metrics = {
            "val_loss": 0.52,
            "val_accuracy": 0.83,
            "val_f1": 0.81
        }
        
        logger.log_validation_metrics(metrics, step=1)
        
        assert logger.metrics_buffer["val_loss"] == [(1, 0.52)]
        assert logger.metrics_buffer["val_accuracy"] == [(1, 0.83)]
        assert logger.metrics_buffer["val_f1"] == [(1, 0.81)]
    
    def test_flush_to_mlflow(self):
        """Test flushing buffered metrics to MLflow."""
        from src.mlops.experiment_tracking import MetricsLogger
        
        logger = MetricsLogger()
        
        # Add metrics to buffer
        logger.log_training_metrics({"train_loss": 0.45}, step=1)
        logger.log_validation_metrics({"val_loss": 0.52}, step=1)
        
        # Mock MLflow client
        mock_client = Mock()
        
        logger.flush_to_mlflow(mock_client, run_id="test_run_id")
        
        # Verify metrics were logged to MLflow
        expected_calls = [
            (("test_run_id", "train_loss", 0.45, None, 1),),
            (("test_run_id", "val_loss", 0.52, None, 1),)
        ]
        
        actual_calls = [call.args for call in mock_client.log_metric.call_args_list]
        assert len(actual_calls) == 2
        
        # Buffer should be cleared after flush
        assert logger.metrics_buffer == {}


class TestArtifactManager:
    """Test cases for artifact management functionality."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
    
    def teardown_method(self):
        """Clean up after tests."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_artifact_manager_initialization(self):
        """Test ArtifactManager initialization."""
        from src.mlops.experiment_tracking import ArtifactManager
        
        manager = ArtifactManager(base_path=self.temp_dir)
        
        assert manager.base_path == Path(self.temp_dir)
        assert manager.base_path.exists()
    
    def test_save_model_checkpoint(self):
        """Test saving model checkpoints with metadata."""
        from src.mlops.experiment_tracking import ArtifactManager
        
        manager = ArtifactManager(base_path=self.temp_dir)
        
        # Mock model
        mock_model = Mock()
        mock_model.state_dict.return_value = {"layer.weight": "fake_tensor"}
        
        checkpoint_info = {
            "epoch": 5,
            "step": 1000,
            "loss": 0.23,
            "accuracy": 0.87
        }
        
        checkpoint_path = manager.save_model_checkpoint(
            model=mock_model,
            checkpoint_info=checkpoint_info,
            filename="checkpoint_epoch_5.pt"
        )
        
        assert checkpoint_path.exists()
        assert checkpoint_path.name == "checkpoint_epoch_5.pt"
        
        # Verify metadata file was created
        metadata_path = checkpoint_path.parent / f"{checkpoint_path.stem}_metadata.json"
        assert metadata_path.exists()
    
    def test_save_training_logs(self):
        """Test saving training logs."""
        from src.mlops.experiment_tracking import ArtifactManager
        
        manager = ArtifactManager(base_path=self.temp_dir)
        
        logs = [
            {"epoch": 1, "step": 100, "loss": 0.8, "accuracy": 0.7},
            {"epoch": 1, "step": 200, "loss": 0.6, "accuracy": 0.8},
            {"epoch": 2, "step": 300, "loss": 0.4, "accuracy": 0.85}
        ]
        
        log_path = manager.save_training_logs(logs, filename="training_log.json")
        
        assert log_path.exists()
        assert log_path.name == "training_log.json"
    
    def test_save_config(self):
        """Test saving configuration files."""
        from src.mlops.experiment_tracking import ArtifactManager
        
        manager = ArtifactManager(base_path=self.temp_dir)
        
        config = {
            "model": {
                "name": "bert-base-uncased",
                "num_labels": 2
            },
            "training": {
                "learning_rate": 0.001,
                "batch_size": 32,
                "epochs": 3
            }
        }
        
        config_path = manager.save_config(config, filename="experiment_config.json")
        
        assert config_path.exists()
        assert config_path.name == "experiment_config.json"
    
    def test_create_artifact_directory(self):
        """Test creating artifact directories with proper structure."""
        from src.mlops.experiment_tracking import ArtifactManager
        
        manager = ArtifactManager(base_path=self.temp_dir)
        
        artifact_dir = manager.create_artifact_directory("experiment_001")
        
        expected_subdirs = ["models", "configs", "logs", "plots", "data"]
        
        assert artifact_dir.exists()
        for subdir in expected_subdirs:
            assert (artifact_dir / subdir).exists()
            assert (artifact_dir / subdir).is_dir()