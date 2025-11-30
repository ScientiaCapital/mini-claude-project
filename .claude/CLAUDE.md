# CLAUDE.md - Mini-Claude Project Guide

## Project Status & Overview

**Mini-Claude** is an **educational AI/ML project** focused on building a functional chatbot while learning transformer architecture and LoRA fine-tuning. The project follows **Test-Driven Development (TDD)** principles and is currently in **active development**.

**Current Focus:**
- 🤖 MVP chatbot implementation with pre-trained models
- 🧠 Transformer architecture understanding
- 🔧 LoRA fine-tuning experimentation
- 🎨 Gradio web interface development

## Technology Stack

### Core Framework & Language
- **Language**: Python 3.8+
- **Web Framework**: FastAPI
- **Testing**: pytest ecosystem
- **AI/ML**: PyTorch, Transformers

### Key Dependencies
```python
# Core AI/ML
torch                    # Deep learning framework
transformers             # Pre-trained models
peft                     # LoRA fine-tuning
datasets                 # HuggingFace datasets

# Web & API
fastapi                  # Web framework
uvicorn                  # ASGI server
gradio                   # Web interface

# Testing
pytest                   # Test framework
pytest-cov               # Coverage reporting
pytest-timeout           # Test timeouts
pytest-asyncio           # Async test support

# Development
black, isort, flake8     # Code quality
pre-commit               # Git hooks
```

## Development Workflow

### Initial Setup
```bash
# 1. Clone and setup environment
git clone https://github.com/yourusername/mini-claude.git
cd mini-claude
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows

# 2. Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt  # Development dependencies

# 3. Download models (first time)
huggingface-cli download microsoft/DialoGPT-medium
# OR use our script
python scripts/download_models.py
```

### Running the Application

**MVP Chatbot (CLI):**
```bash
python src/mvp_chatbot.py
```

**Web Interface:**
```bash
python src/web_app.py
```

**FastAPI Development Server:**
```bash
uvicorn src.web.api:app --reload --port 8000
```

### Testing Workflow (TDD Approach)
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test categories
pytest tests/unit/           # Unit tests only
pytest tests/integration/    # Integration tests only
pytest tests/ -k "test_chat" # Tests with "chat" in name

# Run with timeout protection
pytest --timeout=30

# Watch mode (requires pytest-watch)
ptw
```

## Environment Variables

Create a `.env` file in the project root:

```env
# Model Configuration
MODEL_NAME=microsoft/DialoGPT-medium
MODEL_CACHE_DIR=./models
MAX_LENGTH=512

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
DEBUG=True

# Training Configuration
BATCH_SIZE=4
LEARNING_RATE=2e-5
LORA_RANK=16

# Optional: HuggingFace Token (for private models)
HUGGINGFACE_TOKEN=your_token_here
```

## Key Files & Their Purposes

### Core Implementation
- `src/mvp_chatbot.py` - CLI chatbot entry point
- `src/web_app.py` - Gradio web interface
- `src/web/api.py` - FastAPI endpoints
- `src/models/transformer.py` - Custom transformer implementation
- `src/models/lora_wrapper.py` - LoRA fine-tuning wrapper
- `src/training/trainer.py` - Training loop and utilities

### Testing Structure
- `tests/unit/test_models.py` - Unit tests for model components
- `tests/unit/test_chatbot.py` - Chatbot functionality tests
- `tests/integration/test_api.py` - API integration tests
- `tests/integration/test_training.py` - Training pipeline tests
- `tests/conftest.py` - pytest fixtures and configuration

### Configuration & Scripts
- `scripts/download_models.py` - Model downloading utility
- `scripts/setup_training.py` - Training environment setup
- `notebooks/` - Jupyter notebooks for experimentation
- `resources/repos/` - Cloned educational repositories

## Testing Approach

### TDD Philosophy
This project strictly follows **Test-Driven Development**:
1. Write failing tests first
2. Implement minimal code to pass tests
3. Refactor while maintaining test coverage

### Test Categories
```python
# Unit Tests - Fast, isolated
def test_transformer_forward_pass():
    """Test transformer model forward pass"""
    # Arrange
    model = TransformerModel()
    input_ids = torch.tensor([[1, 2, 3]])
    
    # Act
    output = model(input_ids)
    
    # Assert
    assert output.shape == expected_shape

# Integration Tests - Component interaction
def test_chatbot_pipeline():
    """Test complete chatbot pipeline"""
    # Test model loading → preprocessing → generation → response

# Async Tests - For FastAPI endpoints
@pytest.mark.asyncio
async def test_chat_endpoint():
    """Test async chat endpoint"""
    response = await client.post("/chat", json={"message": "Hello"})
    assert response.status_code == 200
```

### Test Configuration
```python
# pytest configuration in pyproject.toml or pytest.ini
[tool.pytest.ini_options]
testpaths = ["tests"]
asyncio_mode = "auto"
timeout = 30
addopts = "--strict-markers --strict-config"
markers = [
    "slow: marks tests as slow (deselect with '-m \"not slow\"')",
    "integration: integration tests",
    "unit: unit tests"
]
```

## Deployment Strategy

### Current Stage: Development
- **Local development** with hot-reload
- **Model caching** to avoid re-downloads
- **Gradio sharing** for demo purposes

### Future Production Considerations
```python
# Planned deployment options
1. HuggingFace Spaces (Gradio)
2. FastAPI + Uvicorn on cloud VM
3. Docker containerization (future)
4. Model quantization for reduced memory
```

## Coding Standards

### Python & FastAPI Specifics
```python
# Use type hints throughout
def process_message(
    message: str,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer
) -> Dict[str, Any]:
    """Process chat message with type hints."""
    
# Async/await pattern for FastAPI
@app.post("/chat")
async def chat_endpoint(chat_request: ChatRequest) -> ChatResponse:
    """Async endpoint for chat functionality."""
    return await chat_service.process(chat_request)

# Error handling with specific exceptions
class ModelLoadingError(Exception):
    """Custom exception for model loading failures."""

# Configuration management with Pydantic
class ModelConfig(BaseModel):
    model_name: str = "microsoft/DialoGPT-medium"
    max_length: int = Field(512, ge=1, le=1024)
```

### Project-Specific Conventions
- **File naming**: `snake_case` for all files
- **Test naming**: `test_*` for test files and functions
- **Model classes**: Suffix with `Model` (e.g., `ChatModel`)
- **Service classes**: Suffix with `Service` (e.g., `TrainingService`)

## Common Tasks & Commands

### Development Commands
```bash
# Code quality
black src/ tests/                    # Auto-formatting
isort src/ tests/                    # Import sorting
flake8 src/ tests/                   # Linting

# Pre-commit hooks (if configured)
pre-commit run --all-files

# Model management
python scripts/download_models.py --model microsoft/DialoGPT-medium
python scripts/clean_cache.py        # Clean model cache

# Training experiments
python src/training/train_lora.py --config configs/lora_default.yaml
```

### Testing Commands
```bash
# Common test patterns
pytest -x                           # Stop on first failure
pytest -v                           # Verbose output
pytest --lf                         # Run last failed
pytest -m "not slow"               # Skip slow tests
pytest --cov=src --cov-report=term-missing  # Coverage with missing lines

# Specific test targets
pytest tests/unit/test_models.py::test_transformer_attention -v
pytest tests/integration/ -k "api"  # Integration API tests
```

### Debugging Commands
```bash
# Debug model loading
python -c "from transformers import AutoModel; model = AutoModel.from_pretrained('microsoft/DialoGPT-medium')"

# Check GPU availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Test tokenizer
python scripts/test_tokenizer.py --text "Hello, world!"
```

## Troubleshooting Tips

### Common Issues & Solutions

**Model Download Failures:**
```bash
# Set cache directory
export HF_HOME=./models
# Or use offline mode if previously downloaded
python scripts/download_models.py --offline
```

**GPU Memory Issues:**
```python
# Reduce batch size in training
config.batch_size = 2
# Use gradient accumulation
config.gradient_accumulation_steps = 4
# Enable memory efficient attention
model.config.use_memory_efficient_attention = True
```

**Test Timeouts:**
```python
# Mark slow tests
@pytest.mark.slow
def test_training_convergence():
    # This test might take minutes
    pass

# Run without slow tests
pytest -m "not slow"
```

**FastAPI Async Issues:**
```python
# Ensure proper async handling
@app.post("/chat")
async def chat_endpoint(request: Request):
    # Use async/await for IO operations
    data = await request.json()
    return await process_async(data)
```

### Performance Optimization
```python
# Enable torch compile for faster inference (PyTorch 2.0+)
model = torch.compile(model)

# Use half precision for inference
model.half()

# Implement response caching for frequent queries
from functools import lru_cache

@lru_cache(maxsize=1000)
def get_cached_response(message: str) -> str:
    return model.generate(message)
```

This CLAUDE.md provides comprehensive guidance specific to the Mini-Claude project's educational AI/ML focus, FastAPI framework, and TDD approach. Update it as the project evolves through the 12-week learning journey.