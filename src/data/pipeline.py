"""
Data Processing Pipeline
Complete pipeline for processing training data with checkpointing and quality control.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable, Tuple
import logging
import time
import hashlib
from dataclasses import dataclass, asdict
from datetime import datetime

from .quality_metrics import DataQualityAssessment
from .alpaca_utils import AlpacaDataset, load_alpaca_dataset, save_alpaca_dataset
from .sample_generator import generate_sample_dataset

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Configuration for data processing pipeline."""
    input_path: str
    output_path: str
    checkpoint_dir: str = "checkpoints"
    quality_threshold: float = 0.7
    batch_size: int = 1000
    enable_checkpointing: bool = True
    max_retries: int = 3
    validation_split: float = 0.1
    random_seed: int = 42


@dataclass
class PipelineStep:
    """Definition of a pipeline processing step."""
    name: str
    function: Callable
    description: str
    required: bool = True
    checkpoint_name: Optional[str] = None


@dataclass
class PipelineState:
    """State tracking for pipeline execution."""
    config: PipelineConfig
    current_step: int = 0
    completed_steps: List[str] = None
    start_time: datetime = None
    last_checkpoint: Optional[str] = None
    
    def __post_init__(self):
        if self.completed_steps is None:
            self.completed_steps = []
        if self.start_time is None:
            self.start_time = datetime.now()


class DataProcessingPipeline:
    """
    Main data processing pipeline with quality control and checkpointing.
    """
    
    def __init__(self, config: PipelineConfig):
        """
        Initialize the pipeline.
        
        Args:
            config: Pipeline configuration
        """
        self.config = config
        self.state = PipelineState(config=config)
        self.quality_assessor = DataQualityAssessment()
        
        # Create directories
        Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        Path(config.output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Define pipeline steps
        self.steps = [
            PipelineStep(
                name="load_data",
                function=self._load_data,
                description="Load and validate input data",
                checkpoint_name="data_loaded"
            ),
            PipelineStep(
                name="initial_quality_check",
                function=self._initial_quality_check,
                description="Perform initial quality assessment",
                checkpoint_name="quality_checked"
            ),
            PipelineStep(
                name="clean_data",
                function=self._clean_data,
                description="Clean and preprocess data",
                checkpoint_name="data_cleaned"
            ),
            PipelineStep(
                name="augment_data",
                function=self._augment_data,
                description="Augment dataset if needed",
                required=False,
                checkpoint_name="data_augmented"
            ),
            PipelineStep(
                name="final_quality_check",
                function=self._final_quality_check,
                description="Perform final quality validation",
                checkpoint_name="final_quality_checked"
            ),
            PipelineStep(
                name="split_data",
                function=self._split_data,
                description="Split into train/validation sets",
                checkpoint_name="data_split"
            ),
            PipelineStep(
                name="save_data",
                function=self._save_data,
                description="Save processed data",
                checkpoint_name="data_saved"
            )
        ]
        
        logger.info(f"Pipeline initialized with {len(self.steps)} steps")
    
    def run(self, resume_from_checkpoint: bool = True) -> Dict[str, Any]:
        """
        Run the complete pipeline.
        
        Args:
            resume_from_checkpoint: Whether to resume from last checkpoint
            
        Returns:
            Pipeline execution report
        """
        logger.info("Starting data processing pipeline")
        start_time = time.time()
        
        # Resume from checkpoint if enabled
        if resume_from_checkpoint and self.config.enable_checkpointing:
            self._load_checkpoint()
        
        results = {}
        
        try:
            # Execute pipeline steps
            for i, step in enumerate(self.steps[self.state.current_step:], 
                                   start=self.state.current_step):
                
                self.state.current_step = i
                
                logger.info(f"Executing step {i+1}/{len(self.steps)}: {step.name}")
                logger.info(f"Description: {step.description}")
                
                # Execute step with retries
                step_result = self._execute_step_with_retries(step)
                results[step.name] = step_result
                
                # Mark step as completed
                self.state.completed_steps.append(step.name)
                
                # Create checkpoint if enabled
                if self.config.enable_checkpointing and step.checkpoint_name:
                    self._create_checkpoint(step.checkpoint_name, results)
                
                logger.info(f"Completed step: {step.name}")
            
            # Final report
            execution_time = time.time() - start_time
            
            final_report = {
                "status": "completed",
                "execution_time_seconds": execution_time,
                "completed_steps": self.state.completed_steps,
                "step_results": results,
                "config": asdict(self.config),
                "final_metrics": results.get("final_quality_check", {})
            }
            
            logger.info(f"Pipeline completed successfully in {execution_time:.2f} seconds")
            
            return final_report
            
        except Exception as e:
            logger.error(f"Pipeline failed at step {self.state.current_step}: {e}")
            
            # Save error checkpoint
            if self.config.enable_checkpointing:
                error_checkpoint = {
                    "error": str(e),
                    "failed_step": self.steps[self.state.current_step].name,
                    "completed_steps": self.state.completed_steps,
                    "partial_results": results
                }
                self._create_checkpoint("error", error_checkpoint)
            
            raise
    
    def _execute_step_with_retries(self, step: PipelineStep) -> Any:
        """Execute a pipeline step with retry logic."""
        for attempt in range(self.config.max_retries):
            try:
                return step.function()
            except Exception as e:
                if attempt == self.config.max_retries - 1:
                    raise
                
                logger.warning(f"Step {step.name} failed (attempt {attempt + 1}): {e}")
                logger.info(f"Retrying in 2 seconds...")
                time.sleep(2)
    
    def _load_data(self) -> Dict[str, Any]:
        """Load and validate input data."""
        logger.info(f"Loading data from: {self.config.input_path}")
        
        input_path = Path(self.config.input_path)
        
        if not input_path.exists():
            # Generate sample data if input doesn't exist
            logger.warning(f"Input file not found. Generating sample data...")
            dataset = generate_sample_dataset(num_samples=100, random_seed=self.config.random_seed)
            save_alpaca_dataset(dataset, input_path)
        
        # Load the dataset
        self.dataset = load_alpaca_dataset(input_path)
        
        return {
            "records_loaded": len(self.dataset.data),
            "input_path": str(input_path),
            "metadata": self.dataset.metadata
        }
    
    def _initial_quality_check(self) -> Dict[str, Any]:
        """Perform initial quality assessment."""
        logger.info("Performing initial quality assessment")
        
        # Convert to DataFrame for quality assessment
        df = pd.DataFrame([record.dict() for record in self.dataset.data])
        
        # Run quality assessment
        quality_report = self.quality_assessor.assess_dataset(df)
        
        # Check if quality meets threshold
        if quality_report['overall_score'] < self.config.quality_threshold:
            logger.warning(
                f"Initial quality score ({quality_report['overall_score']:.3f}) "
                f"below threshold ({self.config.quality_threshold})"
            )
        
        return quality_report
    
    def _clean_data(self) -> Dict[str, Any]:
        """Clean and preprocess data."""
        logger.info("Cleaning and preprocessing data")
        
        initial_count = len(self.dataset.data)
        cleaned_data = []
        
        for record in self.dataset.data:
            # Clean whitespace
            record.instruction = record.instruction.strip()
            record.input = record.input.strip()
            record.output = record.output.strip()
            
            # Remove records with very short outputs (likely low quality)
            if len(record.output) < 10:
                continue
            
            # Remove records with very long outputs (potential spam/noise)
            if len(record.output) > 5000:
                continue
            
            # Basic content filtering
            if self._is_valid_content(record):
                cleaned_data.append(record)
        
        # Update dataset
        self.dataset.data = cleaned_data
        
        removed_count = initial_count - len(cleaned_data)
        
        return {
            "initial_count": initial_count,
            "final_count": len(cleaned_data),
            "removed_count": removed_count,
            "removal_rate": removed_count / initial_count if initial_count > 0 else 0
        }
    
    def _is_valid_content(self, record) -> bool:
        """Check if a record contains valid content."""
        # Check for empty or placeholder content
        placeholders = ["[", "todo", "tbd", "placeholder", "example"]
        
        text_to_check = (record.instruction + " " + record.input + " " + record.output).lower()
        
        # Reject if contains too many placeholders
        placeholder_count = sum(1 for p in placeholders if p in text_to_check)
        if placeholder_count > 1:
            return False
        
        # Basic quality checks
        if len(record.instruction.split()) < 3:  # Too short instruction
            return False
        
        if len(record.output.split()) < 5:  # Too short output
            return False
        
        return True
    
    def _augment_data(self) -> Dict[str, Any]:
        """Augment dataset if needed."""
        logger.info("Checking if data augmentation is needed")
        
        current_size = len(self.dataset.data)
        target_size = 500  # Target dataset size
        
        if current_size >= target_size:
            logger.info(f"Dataset size ({current_size}) meets target, skipping augmentation")
            return {"augmented": False, "original_size": current_size}
        
        # Generate additional samples
        needed_samples = target_size - current_size
        logger.info(f"Generating {needed_samples} additional samples")
        
        additional_dataset = generate_sample_dataset(
            num_samples=needed_samples,
            random_seed=self.config.random_seed + 1000
        )
        
        # Merge datasets
        self.dataset.data.extend(additional_dataset.data)
        
        return {
            "augmented": True,
            "original_size": current_size,
            "added_samples": needed_samples,
            "final_size": len(self.dataset.data)
        }
    
    def _final_quality_check(self) -> Dict[str, Any]:
        """Perform final quality validation."""
        logger.info("Performing final quality check")
        
        # Convert to DataFrame for quality assessment
        df = pd.DataFrame([record.dict() for record in self.dataset.data])
        
        # Run quality assessment
        quality_report = self.quality_assessor.assess_dataset(df)
        
        # Ensure quality meets threshold
        if quality_report['overall_score'] < self.config.quality_threshold:
            raise ValueError(
                f"Final quality score ({quality_report['overall_score']:.3f}) "
                f"below threshold ({self.config.quality_threshold}). "
                "Pipeline failed quality validation."
            )
        
        logger.info(f"Final quality score: {quality_report['overall_score']:.3f}")
        
        return quality_report
    
    def _split_data(self) -> Dict[str, Any]:
        """Split data into train/validation sets."""
        logger.info(f"Splitting data (validation split: {self.config.validation_split})")
        
        # Import here to avoid circular imports
        from .alpaca_utils import split_alpaca_dataset
        
        train_dataset, val_dataset = split_alpaca_dataset(
            self.dataset,
            train_ratio=1 - self.config.validation_split,
            random_seed=self.config.random_seed
        )
        
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        
        return {
            "total_samples": len(self.dataset.data),
            "train_samples": len(train_dataset.data),
            "validation_samples": len(val_dataset.data),
            "split_ratio": self.config.validation_split
        }
    
    def _save_data(self) -> Dict[str, Any]:
        """Save processed data."""
        logger.info(f"Saving processed data to: {self.config.output_path}")
        
        output_path = Path(self.config.output_path)
        
        # Save full dataset
        save_alpaca_dataset(self.dataset, output_path)
        
        # Save train/validation splits
        train_path = output_path.parent / f"{output_path.stem}_train.json"
        val_path = output_path.parent / f"{output_path.stem}_val.json"
        
        save_alpaca_dataset(self.train_dataset, train_path)
        save_alpaca_dataset(self.val_dataset, val_path)
        
        return {
            "output_path": str(output_path),
            "train_path": str(train_path),
            "validation_path": str(val_path),
            "total_samples": len(self.dataset.data)
        }
    
    def _create_checkpoint(self, checkpoint_name: str, data: Any) -> None:
        """Create a checkpoint with current state."""
        checkpoint_path = Path(self.config.checkpoint_dir) / f"{checkpoint_name}.json"
        
        checkpoint_data = {
            "checkpoint_name": checkpoint_name,
            "timestamp": datetime.now().isoformat(),
            "state": asdict(self.state),
            "data": data,
            "config_hash": self._get_config_hash()
        }
        
        with open(checkpoint_path, 'w') as f:
            json.dump(checkpoint_data, f, indent=2, default=str)
        
        self.state.last_checkpoint = checkpoint_name
        logger.info(f"Created checkpoint: {checkpoint_path}")
    
    def _load_checkpoint(self) -> bool:
        """Load the most recent checkpoint."""
        checkpoint_dir = Path(self.config.checkpoint_dir)
        
        if not checkpoint_dir.exists():
            return False
        
        # Find most recent checkpoint
        checkpoints = list(checkpoint_dir.glob("*.json"))
        if not checkpoints:
            return False
        
        latest_checkpoint = max(checkpoints, key=lambda p: p.stat().st_mtime)
        
        try:
            with open(latest_checkpoint) as f:
                checkpoint_data = json.load(f)
            
            # Verify config compatibility
            if checkpoint_data.get("config_hash") != self._get_config_hash():
                logger.warning("Config changed since checkpoint, starting fresh")
                return False
            
            # Restore state
            self.state = PipelineState(**checkpoint_data["state"])
            
            logger.info(f"Resumed from checkpoint: {latest_checkpoint}")
            return True
            
        except Exception as e:
            logger.warning(f"Failed to load checkpoint: {e}")
            return False
    
    def _get_config_hash(self) -> str:
        """Get a hash of the current configuration."""
        config_str = json.dumps(asdict(self.config), sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()


def create_pipeline_from_config(config_path: str) -> DataProcessingPipeline:
    """
    Create a pipeline from a configuration file.
    
    Args:
        config_path: Path to JSON configuration file
        
    Returns:
        Configured DataProcessingPipeline
    """
    with open(config_path) as f:
        config_dict = json.load(f)
    
    config = PipelineConfig(**config_dict)
    return DataProcessingPipeline(config)


if __name__ == "__main__":
    # Example usage
    config = PipelineConfig(
        input_path="data/raw/input.json",
        output_path="data/processed/output.json",
        checkpoint_dir="data/checkpoints",
        quality_threshold=0.7,
        validation_split=0.1
    )
    
    pipeline = DataProcessingPipeline(config)
    report = pipeline.run()
    
    print(f"Pipeline completed: {report['status']}")
    print(f"Processing time: {report['execution_time_seconds']:.2f} seconds")
    print(f"Final quality score: {report['final_metrics']['overall_score']:.3f}")