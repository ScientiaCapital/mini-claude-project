"""
Alpaca Format Utilities
Utilities for working with Alpaca training format data.
"""

import json
import pandas as pd
from typing import Dict, List, Any, Optional, Union
from pydantic import BaseModel, Field, validator
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class AlpacaRecord(BaseModel):
    """
    Pydantic model for validating individual Alpaca format records.
    """
    instruction: str = Field(..., min_length=1, description="The task instruction")
    input: str = Field(default="", description="Additional context or input for the task")
    output: str = Field(..., min_length=1, description="The expected response or output")
    
    @validator('instruction')
    def instruction_not_empty(cls, v):
        if not v.strip():
            raise ValueError('Instruction cannot be empty or whitespace only')
        return v.strip()
    
    @validator('output')
    def output_not_empty(cls, v):
        if not v.strip():
            raise ValueError('Output cannot be empty or whitespace only')
        return v.strip()
    
    @validator('input')
    def clean_input(cls, v):
        return v.strip() if v else ""


class AlpacaDataset(BaseModel):
    """
    Pydantic model for validating complete Alpaca format datasets.
    """
    data: List[AlpacaRecord] = Field(..., min_items=1)
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)
    
    @validator('data')
    def validate_unique_combinations(cls, v):
        """Ensure no duplicate instruction-input combinations."""
        seen = set()
        unique_records = []
        for record in v:
            key = (record.instruction, record.input)
            if key not in seen:
                seen.add(key)
                unique_records.append(record)
        
        if len(unique_records) != len(v):
            logger.warning(f"Removed {len(v) - len(unique_records)} duplicate records")
        
        return unique_records


def load_alpaca_dataset(file_path: Union[str, Path]) -> AlpacaDataset:
    """
    Load and validate an Alpaca format dataset from file.
    
    Args:
        file_path: Path to JSON file containing Alpaca format data
        
    Returns:
        Validated AlpacaDataset object
        
    Raises:
        ValueError: If data format is invalid
        FileNotFoundError: If file doesn't exist
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {file_path}")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
        
        # Handle different input formats
        if isinstance(raw_data, list):
            # Direct list of records
            dataset_data = {"data": raw_data}
        elif isinstance(raw_data, dict) and "data" in raw_data:
            # Wrapped format with metadata
            dataset_data = raw_data
        else:
            raise ValueError("Invalid file format. Expected list of records or dict with 'data' key.")
        
        return AlpacaDataset(**dataset_data)
        
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format: {e}")
    except Exception as e:
        raise ValueError(f"Error loading dataset: {e}")


def save_alpaca_dataset(dataset: AlpacaDataset, file_path: Union[str, Path]) -> None:
    """
    Save an Alpaca dataset to file.
    
    Args:
        dataset: AlpacaDataset object to save
        file_path: Output file path
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert to dict and save
    data_dict = {
        "data": [record.dict() for record in dataset.data],
        "metadata": dataset.metadata
    }
    
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data_dict, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Saved {len(dataset.data)} records to {file_path}")


def alpaca_to_dataframe(dataset: AlpacaDataset) -> pd.DataFrame:
    """
    Convert AlpacaDataset to pandas DataFrame.
    
    Args:
        dataset: AlpacaDataset object
        
    Returns:
        DataFrame with instruction, input, output columns
    """
    records = [record.dict() for record in dataset.data]
    return pd.DataFrame(records)


def dataframe_to_alpaca(df: pd.DataFrame) -> AlpacaDataset:
    """
    Convert pandas DataFrame to AlpacaDataset.
    
    Args:
        df: DataFrame with instruction, input, output columns
        
    Returns:
        AlpacaDataset object
    """
    required_columns = {'instruction', 'output'}
    if not required_columns.issubset(df.columns):
        missing = required_columns - set(df.columns)
        raise ValueError(f"DataFrame missing required columns: {missing}")
    
    # Ensure input column exists
    if 'input' not in df.columns:
        df = df.copy()
        df['input'] = ""
    
    # Convert to list of dicts
    records = df[['instruction', 'input', 'output']].to_dict('records')
    
    return AlpacaDataset(data=records)


def generate_alpaca_template(instruction_type: str = "general") -> Dict[str, str]:
    """
    Generate a template for creating Alpaca format data.
    
    Args:
        instruction_type: Type of instruction template to generate
        
    Returns:
        Template dictionary with instruction, input, output fields
    """
    templates = {
        "general": {
            "instruction": "Complete the following task:",
            "input": "[Provide specific context or input here]",
            "output": "[Expected response or completion here]"
        },
        "qa": {
            "instruction": "Answer the following question:",
            "input": "[Question text here]",
            "output": "[Answer here]"
        },
        "translation": {
            "instruction": "Translate the following text from {source_lang} to {target_lang}:",
            "input": "[Text to translate here]",
            "output": "[Translated text here]"
        },
        "coding": {
            "instruction": "Write a Python function that:",
            "input": "[Function requirements or specification here]",
            "output": "[Python code implementation here]"
        },
        "summarization": {
            "instruction": "Summarize the following text:",
            "input": "[Long text to summarize here]",
            "output": "[Concise summary here]"
        }
    }
    
    return templates.get(instruction_type, templates["general"])


def validate_alpaca_quality(dataset: AlpacaDataset) -> Dict[str, Any]:
    """
    Perform quality validation specific to Alpaca format data.
    
    Args:
        dataset: AlpacaDataset to validate
        
    Returns:
        Quality validation report
    """
    report = {
        "total_records": len(dataset.data),
        "validation_errors": [],
        "quality_metrics": {},
        "recommendations": []
    }
    
    # Length distribution analysis
    instruction_lengths = [len(record.instruction) for record in dataset.data]
    input_lengths = [len(record.input) for record in dataset.data]
    output_lengths = [len(record.output) for record in dataset.data]
    
    report["quality_metrics"]["length_stats"] = {
        "instruction": {
            "mean": sum(instruction_lengths) / len(instruction_lengths),
            "min": min(instruction_lengths),
            "max": max(instruction_lengths)
        },
        "input": {
            "mean": sum(input_lengths) / len(input_lengths),
            "min": min(input_lengths),
            "max": max(input_lengths)
        },
        "output": {
            "mean": sum(output_lengths) / len(output_lengths),
            "min": min(output_lengths),
            "max": max(output_lengths)
        }
    }
    
    # Check for very short outputs (might indicate poor quality)
    short_outputs = [i for i, length in enumerate(output_lengths) if length < 10]
    if short_outputs:
        report["validation_errors"].append(
            f"Found {len(short_outputs)} records with very short outputs (< 10 chars)"
        )
    
    # Check for very long outputs (might indicate verbosity issues)
    long_outputs = [i for i, length in enumerate(output_lengths) if length > 2000]
    if long_outputs:
        report["validation_errors"].append(
            f"Found {len(long_outputs)} records with very long outputs (> 2000 chars)"
        )
    
    # Check for empty inputs (not necessarily bad, but worth noting)
    empty_inputs = sum(1 for record in dataset.data if not record.input.strip())
    report["quality_metrics"]["empty_input_ratio"] = empty_inputs / len(dataset.data)
    
    # Generate recommendations
    if report["quality_metrics"]["empty_input_ratio"] > 0.5:
        report["recommendations"].append(
            "High ratio of empty inputs. Consider if context is needed for better training."
        )
    
    if short_outputs:
        report["recommendations"].append(
            "Some outputs are very short. Review for completeness and quality."
        )
    
    if long_outputs:
        report["recommendations"].append(
            "Some outputs are very long. Consider truncation or breaking into smaller examples."
        )
    
    report["is_valid"] = len(report["validation_errors"]) == 0
    
    return report


def merge_alpaca_datasets(*datasets: AlpacaDataset) -> AlpacaDataset:
    """
    Merge multiple Alpaca datasets into one.
    
    Args:
        *datasets: AlpacaDataset objects to merge
        
    Returns:
        Merged AlpacaDataset
    """
    all_records = []
    merged_metadata = {}
    
    for i, dataset in enumerate(datasets):
        all_records.extend(dataset.data)
        merged_metadata[f"source_{i}"] = {
            "record_count": len(dataset.data),
            "metadata": dataset.metadata
        }
    
    return AlpacaDataset(
        data=all_records,
        metadata={
            "merged_from": len(datasets),
            "total_records": len(all_records),
            "sources": merged_metadata
        }
    )


def split_alpaca_dataset(dataset: AlpacaDataset, 
                        train_ratio: float = 0.8,
                        random_seed: int = 42) -> tuple[AlpacaDataset, AlpacaDataset]:
    """
    Split an Alpaca dataset into train and test sets.
    
    Args:
        dataset: AlpacaDataset to split
        train_ratio: Ratio of data for training (0.0 to 1.0)
        random_seed: Random seed for reproducible splits
        
    Returns:
        Tuple of (train_dataset, test_dataset)
    """
    import random
    
    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be between 0 and 1")
    
    # Create a copy of the data and shuffle
    data_copy = dataset.data.copy()
    random.seed(random_seed)
    random.shuffle(data_copy)
    
    # Calculate split index
    split_idx = int(len(data_copy) * train_ratio)
    
    train_data = data_copy[:split_idx]
    test_data = data_copy[split_idx:]
    
    train_dataset = AlpacaDataset(
        data=train_data,
        metadata={
            **dataset.metadata,
            "split": "train",
            "split_ratio": train_ratio,
            "random_seed": random_seed
        }
    )
    
    test_dataset = AlpacaDataset(
        data=test_data,
        metadata={
            **dataset.metadata,
            "split": "test",
            "split_ratio": 1 - train_ratio,
            "random_seed": random_seed
        }
    )
    
    return train_dataset, test_dataset