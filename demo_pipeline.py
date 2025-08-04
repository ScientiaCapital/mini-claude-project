#!/usr/bin/env python3
"""
Training Data Pipeline Demo
Demonstrates the complete training data preparation pipeline for transformer learning.
"""

import sys
import logging
from pathlib import Path

# Add src to path for imports
sys.path.append('.')

from src.data.pipeline import DataProcessingPipeline, PipelineConfig
from src.data.quality_metrics import DataQualityAssessment
from src.data.sample_generator import generate_sample_dataset, create_balanced_dataset
from src.data.alpaca_utils import load_alpaca_dataset, save_alpaca_dataset

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Run the complete training data pipeline demonstration."""
    print("=" * 80)
    print("TRAINING DATA PIPELINE FOR TRANSFORMER LEARNING")
    print("=" * 80)
    
    # Step 1: Generate sample datasets
    print("\n1. GENERATING SAMPLE DATASETS")
    print("-" * 40)
    
    output_dir = Path('data/synthetic')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate different types of datasets
    datasets = {
        'demo_conversational': generate_sample_dataset(
            num_samples=25, 
            random_seed=42
        ),
        'demo_balanced': create_balanced_dataset(
            samples_per_domain=3,
            random_seed=42
        ),
    }
    
    for name, dataset in datasets.items():
        output_path = output_dir / f'{name}.json'
        save_alpaca_dataset(dataset, output_path)
        print(f"✓ Generated {name}: {len(dataset.data)} samples → {output_path}")
    
    # Step 2: Quality Assessment
    print("\n2. DATA QUALITY ASSESSMENT")
    print("-" * 40)
    
    assessor = DataQualityAssessment()
    
    for name, dataset in datasets.items():
        # Convert to DataFrame for assessment
        import pandas as pd
        df = pd.DataFrame([record.dict() for record in dataset.data])
        
        quality_report = assessor.assess_dataset(df)
        print(f"\n{name.upper()} QUALITY REPORT:")
        print(f"  Overall Score: {quality_report['overall_score']:.3f}")
        print(f"  Completeness: {quality_report['completeness']['overall']:.3f}")
        print(f"  Format Valid: {quality_report['format_validation']['is_valid']}")
        print(f"  Sample Count: {len(dataset.data)}")
        
        if quality_report['recommendations']:
            print("  Recommendations:")
            for rec in quality_report['recommendations'][:2]:  # Show first 2
                print(f"    - {rec}")
    
    # Step 3: Data Processing Pipeline
    print("\n3. DATA PROCESSING PIPELINE")
    print("-" * 40)
    
    # Configure pipeline
    config = PipelineConfig(
        input_path='data/synthetic/demo_conversational.json',
        output_path='data/processed/demo_final.json',
        checkpoint_dir='data/checkpoints',
        quality_threshold=0.7,
        validation_split=0.2,
        random_seed=42,
        enable_checkpointing=False  # Simplified for demo
    )
    
    # Run pipeline
    pipeline = DataProcessingPipeline(config)
    report = pipeline.run(resume_from_checkpoint=False)
    
    print(f"✓ Pipeline Status: {report['status']}")
    print(f"✓ Execution Time: {report['execution_time_seconds']:.2f} seconds")
    print(f"✓ Completed Steps: {len(report['completed_steps'])}")
    print(f"✓ Final Quality Score: {report['final_metrics']['overall_score']:.3f}")
    
    # Step 4: Final Dataset Analysis
    print("\n4. FINAL DATASET ANALYSIS")
    print("-" * 40)
    
    # Load and analyze final datasets
    final_dataset = load_alpaca_dataset('data/processed/demo_final.json')
    train_dataset = load_alpaca_dataset('data/processed/demo_final_train.json')
    val_dataset = load_alpaca_dataset('data/processed/demo_final_val.json')
    
    print(f"✓ Total samples: {len(final_dataset.data)}")
    print(f"✓ Training samples: {len(train_dataset.data)}")
    print(f"✓ Validation samples: {len(val_dataset.data)}")
    print(f"✓ Train/Val split: {len(train_dataset.data)}/{len(val_dataset.data)}")
    
    # Analyze content diversity
    instruction_types = {}
    for record in final_dataset.data:
        first_word = record.instruction.split()[0].lower()
        instruction_types[first_word] = instruction_types.get(first_word, 0) + 1
    
    print(f"✓ Instruction diversity: {len(instruction_types)} unique patterns")
    
    # Step 5: Sample Output
    print("\n5. SAMPLE TRAINING EXAMPLES")
    print("-" * 40)
    
    for i, record in enumerate(train_dataset.data[:2]):
        print(f"\nExample {i+1}:")
        print(f"Instruction: {record.instruction}")
        print(f"Input: {record.input}")
        print(f"Output: {record.output[:100]}...")
    
    # Summary
    print("\n" + "=" * 80)
    print("PIPELINE SUMMARY")
    print("=" * 80)
    print(f"✓ Data pipeline architecture created")
    print(f"✓ Quality metrics implemented with TDD approach")
    print(f"✓ Alpaca format validation and utilities ready")
    print(f"✓ Sample conversational dataset generated")
    print(f"✓ Processing pipeline with checkpointing complete")
    print(f"✓ Comprehensive test suite available")
    print(f"\nFinal dataset ready for transformer training!")
    print(f"Training data: data/processed/demo_final_train.json")
    print(f"Validation data: data/processed/demo_final_val.json")
    

if __name__ == "__main__":
    main()