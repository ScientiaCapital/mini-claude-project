# Hugging Face CLI Guide for Mini-Claude Project

## Overview

The Hugging Face CLI (`huggingface-cli`) is essential for our project to:
- Download models efficiently with caching
- Upload fine-tuned models
- Manage model storage
- Authenticate with Hugging Face Hub

## Installation

Already included in our requirements.txt:
```bash
pip install -U "huggingface_hub[cli]"
```

## Key Commands for Our Project

### 1. Authentication (Optional but Recommended)

```bash
# Login to Hugging Face (needed for private models or uploads)
huggingface-cli login

# Check who you're logged in as
huggingface-cli whoami

# Logout
huggingface-cli logout
```

### 2. Downloading Models

```bash
# Download specific model files
huggingface-cli download microsoft/DialoGPT-medium --include "*.json" "*.txt"

# Download entire model
huggingface-cli download microsoft/DialoGPT-medium --local-dir ./models/dialogpt-medium

# Download with specific revision/branch
huggingface-cli download microsoft/DialoGPT-medium --revision main

# Download LoRA adapters
huggingface-cli download tloen/alpaca-lora-7b --local-dir ./models/alpaca-lora
```

### 3. Managing Cache

```bash
# Scan cache to see what's downloaded
huggingface-cli scan-cache

# See detailed cache info
huggingface-cli scan-cache -v

# Delete specific models from cache
huggingface-cli delete-cache --disable-tui microsoft/DialoGPT-small

# Clean up old revisions
huggingface-cli delete-cache --disable-tui --revisions
```

### 4. Uploading Models

```bash
# Create a new model repository
huggingface-cli repo create mini-claude-finetuned --type model

# Upload a single file
huggingface-cli upload mini-claude-finetuned ./models/checkpoint-1000/pytorch_model.bin

# Upload entire folder
huggingface-cli upload mini-claude-finetuned ./models/checkpoint-1000 --repo-type model

# Upload with commit message
huggingface-cli upload mini-claude-finetuned ./models/checkpoint-1000 \
    --commit-message "Add LoRA fine-tuned checkpoint"
```

### 5. Environment Information

```bash
# Check HF environment setup
huggingface-cli env

# This shows:
# - huggingface_hub version
# - Platform details
# - Python version
# - PyTorch/TensorFlow versions
# - Cache directory location
# - Token information
```

## Integration with Our Code

### Using in Python Scripts

```python
from huggingface_hub import hf_hub_download, snapshot_download

# Download a single file
model_file = hf_hub_download(
    repo_id="microsoft/DialoGPT-medium",
    filename="pytorch_model.bin",
    cache_dir="./models"
)

# Download entire model
model_path = snapshot_download(
    repo_id="microsoft/DialoGPT-medium",
    cache_dir="./models",
    local_dir="./models/dialogpt-medium"
)
```

### Environment Variables

Set these in `.env` for the project:
```bash
# Cache directory (default: ~/.cache/huggingface)
HF_HOME=/path/to/cache

# Token for private repos
HUGGING_FACE_HUB_TOKEN=hf_xxxxx

# Offline mode (use only cached models)
HF_HUB_OFFLINE=1

# Disable telemetry
HF_HUB_DISABLE_TELEMETRY=1
```

## Best Practices for Our Project

1. **Pre-download Models**: 
   ```bash
   # Run this after cloning the project
   huggingface-cli download microsoft/DialoGPT-medium
   huggingface-cli download microsoft/DialoGPT-small
   ```

2. **Check Cache Before Training**:
   ```bash
   huggingface-cli scan-cache | grep -E "DialoGPT|gpt2"
   ```

3. **Upload Fine-tuned Models**:
   ```bash
   # After LoRA fine-tuning
   huggingface-cli upload username/mini-claude-lora ./outputs/checkpoint-final
   ```

4. **Manage Disk Space**:
   ```bash
   # Regular cleanup of old model revisions
   huggingface-cli scan-cache --sort-by size
   huggingface-cli delete-cache --disable-tui
   ```

## Common Issues and Solutions

### Issue: "No space left on device"
```bash
# Check cache size
huggingface-cli scan-cache

# Clear unused models
huggingface-cli delete-cache --disable-tui
```

### Issue: "401 Unauthorized"
```bash
# Re-authenticate
huggingface-cli login
```

### Issue: Slow downloads
```bash
# Use local-dir to avoid symlinks
huggingface-cli download model-name --local-dir ./models/model-name

# Resume interrupted downloads (automatic)
huggingface-cli download model-name --resume-download
```

## For CI/CD Integration

```bash
# Set token via environment variable
export HUGGING_FACE_HUB_TOKEN=hf_xxxxx

# Download models in CI pipeline
huggingface-cli download microsoft/DialoGPT-medium --quiet

# Upload results after training
huggingface-cli upload my-org/model-results ./results --quiet
```

This CLI tool will be essential as we work with different models, fine-tune them, and share our results with the community!