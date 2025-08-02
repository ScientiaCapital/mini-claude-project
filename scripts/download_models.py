#!/usr/bin/env python3
"""
Script to download required models for the Mini-Claude project.
This ensures models are cached before running the main application.
"""
import os
import sys
from huggingface_hub import snapshot_download
from tqdm import tqdm


MODELS_TO_DOWNLOAD = [
    {
        "repo_id": "microsoft/DialoGPT-small",
        "local_dir": "models/dialogpt-small",
        "description": "Small DialoGPT model for testing (117M params)"
    },
    {
        "repo_id": "microsoft/DialoGPT-medium", 
        "local_dir": "models/dialogpt-medium",
        "description": "Medium DialoGPT model for production (345M params)"
    },
    # Add more models as needed
    # {
    #     "repo_id": "gpt2",
    #     "local_dir": "models/gpt2",
    #     "description": "Base GPT-2 model for learning (124M params)"
    # },
]


def download_model(repo_id, local_dir, description):
    """Download a model from Hugging Face Hub."""
    print(f"\n📥 Downloading: {description}")
    print(f"   Repository: {repo_id}")
    print(f"   Local path: {local_dir}")
    
    try:
        # Create models directory if it doesn't exist
        os.makedirs(os.path.dirname(local_dir), exist_ok=True)
        
        # Download the model
        snapshot_download(
            repo_id=repo_id,
            local_dir=local_dir,
            local_dir_use_symlinks=False,  # Avoid symlinks for portability
            resume_download=True,  # Resume if interrupted
            # You can add ignore_patterns to skip certain files
            # ignore_patterns=["*.bin"]  # Skip large bin files if only testing
        )
        
        print(f"✅ Successfully downloaded {repo_id}")
        return True
        
    except Exception as e:
        print(f"❌ Failed to download {repo_id}: {e}")
        return False


def main():
    """Main function to download all required models."""
    print("🚀 Mini-Claude Model Downloader")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not os.path.exists("src/mvp_chatbot.py"):
        print("⚠️  Warning: Not in project root directory!")
        print("   Please run from the mini-claude-project directory")
        sys.exit(1)
    
    # Download each model
    success_count = 0
    for model_info in MODELS_TO_DOWNLOAD:
        if download_model(**model_info):
            success_count += 1
    
    # Summary
    print("\n" + "=" * 50)
    print(f"📊 Downloaded {success_count}/{len(MODELS_TO_DOWNLOAD)} models successfully")
    
    if success_count < len(MODELS_TO_DOWNLOAD):
        print("\n⚠️  Some models failed to download.")
        print("   You may need to:")
        print("   1. Check your internet connection")
        print("   2. Run 'huggingface-cli login' if accessing private models")
        print("   3. Check available disk space")
        sys.exit(1)
    else:
        print("\n✅ All models downloaded successfully!")
        print("   You can now run the chatbot offline.")
        
    # Show cache info
    print("\n💾 Cache information:")
    os.system("huggingface-cli scan-cache | grep -E 'DialoGPT|DONE'")


if __name__ == "__main__":
    main()