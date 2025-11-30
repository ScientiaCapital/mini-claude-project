# Mini-Claude Project Architecture

## 1. Technology Stack

### Core Framework
- **FastAPI 0.104+** - Modern, fast web framework for building APIs with Python 3.7+
- **Python 3.8+** - Primary programming language

### AI/ML Stack
- **PyTorch 2.0+** - Deep learning framework
- **Transformers 4.35+** - HuggingFace library for transformer models
- **Datasets 2.14+** - Dataset processing and management
- **Accelerate** - Distributed training utilities
- **PEFT (Parameter-Efficient Fine-Tuning)** - LoRA and other efficient fine-tuning methods
- **bitsandbytes** - 8-bit optimizers and quantization

### Testing & Development
- **pytest** - Testing framework
- **pytest-cov** - Test coverage reporting
- **pytest-timeout** - Test timeout management
- **pytest-asyncio** - Async test support

### Web Interface
- **Gradio** - Web UI framework for ML demos
- **Uvicorn** - ASGI server for FastAPI

## 2. Design Patterns

### Primary Patterns
- **Repository Pattern** - Abstracting data access layer for datasets
- **Strategy Pattern** - Interchangeable model implementations and training strategies
- **Factory Pattern** - Model instantiation and configuration
- **Observer Pattern** - Training progress monitoring and callbacks
- **Dependency Injection** - FastAPI's built-in dependency management

### Architectural Patterns
- **Layered Architecture**:
  - Presentation Layer (FastAPI routes, Gradio interfaces)
  - Service Layer (Business logic, training orchestration)
  - Data Access Layer (Model loading, dataset management)
  - Model Layer (Transformer architectures, fine-tuning logic)

- **Event-Driven Architecture** - For training progress updates and real-time inference

## 3. Key Components

### Core Components
```
src/
├── models/
│   ├── base_model.py          # Abstract base model class
│   ├── dialogpt_wrapper.py    # DialoGPT model implementation
│   └── model_factory.py       # Model instantiation factory
├── training/
│   ├── trainer.py             # Base training logic
│   ├── lora_trainer.py        # LoRA fine-tuning implementation
│   ├── data_processor.py      # Dataset preprocessing
│   └── callbacks.py           # Training callbacks and monitoring
├── web/
│   ├── api.py                 # FastAPI application
│   ├── routes/
│   │   ├── chat.py            # Chat endpoints
│   │   ├── training.py        # Training management endpoints
│   │   └── models.py          # Model management endpoints
│   └── gradio_app.py          # Gradio web interface
└── utils/
    ├── config.py              # Configuration management
    ├── logging.py             # Logging configuration
    └── model_utils.py         # Model utilities and helpers
```

### Component Responsibilities

**Model Layer**
- Model loading and initialization from HuggingFace
- Inference pipeline management
- Model state serialization/deserialization

**Training Layer**
- LoRA configuration and application
- Training loop management
- Gradient accumulation and mixed precision
- Checkpoint management

**Web Layer**
- RESTful API for model interactions
- Real-time chat interface
- Training progress streaming
- Model version management

## 4. Data Flow

### Inference Pipeline
```
User Input → Text Preprocessing → Tokenization → Model Inference → 
Logits Processing → Response Generation → Output Formatting → User Response
```

### Training Pipeline
```
Raw Dataset → Data Loading → Text Preprocessing → Tokenization → 
Batch Creation → Forward Pass → Loss Calculation → Backward Pass → 
Gradient Update → Checkpoint Saving → Metrics Logging
```

### Real-time Chat Flow
```
Web Client → FastAPI Route → Model Service → Inference Engine → 
Response Formatter → Web Client
```

## 5. External Dependencies

### Direct Dependencies
```python
# Core AI/ML
torch>=2.0.0
transformers>=4.35.0
datasets>=2.14.0
accelerate>=0.23.0
peft>=0.5.0
bitsandbytes>=0.41.0

# Web Framework
fastapi>=0.104.0
uvicorn>=0.24.0
gradio>=4.0.0

# Testing
pytest>=7.4.0
pytest-cov>=4.1.0
pytest-timeout>=2.1.0
pytest-asyncio>=0.21.0

# Utilities
numpy>=1.24.0
pydantic>=2.0.0
httpx>=0.25.0
```

### Model Dependencies
- **Primary Model**: microsoft/DialoGPT-medium
- **Tokenizer**: GPT-2 tokenizer compatible
- **Optional Models**: Any HuggingFace transformer model with causal LM head

## 6. API Design

### RESTful Endpoints
```python
# Chat Management
POST /api/v1/chat/completions
GET  /api/v1/chat/sessions/{session_id}
DELETE /api/v1/chat/sessions/{session_id}

# Model Management
GET  /api/v1/models
POST /api/v1/models/load
POST /api/v1/models/unload

# Training Management
POST /api/v1/training/jobs
GET  /api/v1/training/jobs/{job_id}
DELETE /api/v1/training/jobs/{job_id}
GET  /api/v1/training/jobs/{job_id}/progress

# Health & Monitoring
GET  /api/v1/health
GET  /api/v1/metrics
```

### Request/Response Schemas
```python
from pydantic import BaseModel
from typing import List, Optional

class ChatMessage(BaseModel):
    role: str  # "user" or "assistant"
    content: str

class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    max_tokens: Optional[int] = 512
    temperature: Optional[float] = 0.7

class ChatResponse(BaseModel):
    response: str
    tokens_used: int
    processing_time: float

class TrainingJobRequest(BaseModel):
    dataset_path: str
    model_name: str
    lora_config: dict
    training_args: dict
```

### WebSocket Endpoints
- `/ws/chat` - Real-time chat streaming
- `/ws/training/{job_id}` - Training progress streaming

## 7. Database Schema

### In-Memory Storage (No persistent database)
```python
# Session Storage
class ChatSession:
    session_id: str
    user_id: str
    messages: List[ChatMessage]
    created_at: datetime
    last_activity: datetime

# Model Cache
class ModelCache:
    model_id: str
    model_instance: Any
    loaded_at: datetime
    memory_usage: int

# Training Jobs
class TrainingJob:
    job_id: str
    status: str  # "pending", "running", "completed", "failed"
    progress: float
    start_time: datetime
    end_time: Optional[datetime]
    metrics: dict
```

## 8. Security Considerations

### Framework-Specific Security
- **CORS Middleware** - Configured for appropriate origins
- **Rate Limiting** - Implemented via dependencies
- **Input Validation** - Pydantic models for all endpoints
- **SQL Injection** - Not applicable (no SQL database)

### General Security
- **Model Input Sanitization** - Prevent prompt injection attacks
- **Resource Limits** - Memory and compute time restrictions
- **API Key Authentication** - For production deployment
- **Model Access Control** - Restrict model loading capabilities

### Security Dependencies
```python
# Add to requirements for production
python-multipart>=0.0.6  # File upload security
python-jose[cryptography]>=3.3.0  # JWT tokens
passlib[bcrypt]>=1.7.4  # Password hashing
```

## 9. Performance Optimization

### Model Optimization
```python
# Quantization
model = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_threshold=6.0
)

# LoRA Configuration
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# Memory Optimization
model = accelerate.prepare_model(model)
```

### API Performance
- **Async Endpoints** - Non-blocking model inference
- **Response Streaming** - For long-generation tasks
- **Model Caching** - In-memory model instances
- **Connection Pooling** - For external service calls

### Training Optimization
- **Gradient Accumulation** - Effective batch size management
- **Mixed Precision** - FP16 training where supported
- **DataLoader Optimization** - Parallel data loading
- **Checkpoint Strategy** - Smart checkpoint intervals

## 10. Deployment Strategy

### Local Development
```bash
# Development server
uvicorn src.web.api:app --reload --host 0.0.0.0 --port 8000

# Production server (without Docker)
uvicorn src.web.api:app --host 0.0.0.0 --port 8000 --workers 4
```

### Production Considerations
- **Process Management** - Use systemd or supervisord
- **Reverse Proxy** - Nginx for static files and SSL termination
- **Monitoring** - Prometheus metrics endpoint
- **Logging** - Structured JSON logging with rotation

### Scaling Strategy
- **Horizontal Scaling** - Stateless API design
- **Model Serving** - Dedicated model inference servers
- **Load Balancing** - Round-robin for multiple workers
- **Resource Management** - GPU memory monitoring and cleanup

### Future Docker Integration
```dockerfile
# Recommended Dockerfile when added
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["uvicorn", "src.web.api:app", "--host", "0.0.0.0", "--port", "8000"]
```

This architecture provides a solid foundation for an educational AI project while maintaining production-ready patterns and practices. The modular design allows for easy extension and experimentation with different models and training techniques.