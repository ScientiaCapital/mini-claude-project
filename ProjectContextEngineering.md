# ProjectContextEngineering.md

## Technical Context and Engineering Decisions for Mini-Claude

This document captures the technical context, architectural decisions, and engineering rationale for the Mini-Claude project. Updated to reflect the current production state and future learning objectives.

## Current Implementation Status (Updated)

### ✅ **Production Architecture Completed**
**Stack**: Next.js 14 + TypeScript + Neon PostgreSQL + Anthropic Claude + Vercel

**Key Engineering Decisions Made:**
1. **Next.js over Python Flask/FastAPI**: Better TypeScript integration, Vercel deployment
2. **Anthropic Claude over local models**: Higher quality, lower infrastructure complexity
3. **Neon PostgreSQL over SQLite**: Serverless scaling, better production reliability
4. **TDD-first development**: 95%+ test coverage maintained throughout

### 🔄 **Future Learning Components** 
Educational transformer implementation and LoRA fine-tuning remain planned for deep learning understanding.

## Transformer Architecture Context

### Self-Attention Mechanism
Based on insights from `rasbt/LLMs-from-scratch` Chapter 3:

**Implementation Details:**
- Multi-head attention with 12 heads (DialoGPT-medium)
- Attention dimension: 768 (64 per head)
- Scaled dot-product attention: `softmax(QK^T / sqrt(d_k))V`
- Causal masking for autoregressive generation

**Key Design Decisions:**
1. **Pre-normalization**: LayerNorm before attention (more stable training)
2. **Rotary Position Embeddings (RoPE)**: Better length generalization than learned embeddings
3. **Flash Attention**: Optional optimization for longer contexts (requires GPU)

### Positional Encoding Strategy
Following `jaymody/picoGPT` minimal implementation:
```python
# Sinusoidal positional encoding for understanding
# RoPE for production (better extrapolation)
positions = torch.arange(seq_len).unsqueeze(1)
div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
```

### Layer Architecture
Standard transformer block structure:
1. Multi-head self-attention
2. Layer normalization
3. Position-wise feed-forward network (2-layer MLP)
4. Residual connections around both sub-layers

**FFN Expansion Ratio**: 4x (hidden_dim = 4 * model_dim)

## LoRA Fine-tuning Context

### Mathematical Foundation
Low-Rank Adaptation decomposes weight updates:
- Original weights: W ∈ R^(d×k)
- LoRA update: ΔW = BA where B ∈ R^(d×r), A ∈ R^(r×k)
- Rank r << min(d,k), typically r ∈ {8, 16, 32}

### Implementation Strategy
Based on Microsoft's LoRA paper and `huggingface/peft`:

**Target Modules:**
- Query projection (W_q)
- Value projection (W_v)
- Optional: Key projection (W_k) and output projection (W_o)

**Hyperparameters:**
```python
lora_config = {
    "r": 16,                    # Rank
    "lora_alpha": 32,          # Scaling factor
    "lora_dropout": 0.1,       # Dropout for regularization
    "target_modules": ["q_proj", "v_proj"],
    "bias": "none"             # Don't adapt biases
}
```

### Memory Efficiency Analysis
For DialoGPT-medium (345M parameters):
- Full fine-tuning: 345M trainable parameters
- LoRA (r=16): ~0.8M trainable parameters (0.23% of full)
- Memory saving: ~99.77%
- Training speedup: ~10-25x on consumer GPUs

### Training Data Format
Alpaca-style JSON format for compatibility:
```json
{
    "instruction": "You are a helpful AI assistant named Mini-Claude",
    "input": "Hello! How are you today?",
    "output": "Hello! I'm doing well, thank you for asking. How can I help you today?"
}
```

**Minimum Dataset Requirements:**
- MVP: 100 examples (proof of concept)
- Meaningful adaptation: 1,000+ examples
- Production quality: 10,000+ examples

## Model Selection Rationale

### MVP Model: DialoGPT-medium
**Why DialoGPT-medium?**
- Pre-trained on conversational data (147M Reddit conversations)
- Optimal size for learning (345M parameters)
- Runs on CPU with acceptable latency (<2s)
- Good baseline performance without fine-tuning

### Learning Model: GPT-2 small
**Why GPT-2?**
- Well-documented architecture
- Extensive educational resources
- Small enough to train from scratch (124M params)
- Reference implementation in `jaymody/picoGPT`

### Advanced Model: LLaMA-2-7B
**Why LLaMA-2?**
- State-of-the-art open model
- Excellent LoRA support
- Strong instruction-following capabilities
- Active community and tooling

## Dataset Engineering

### Data Collection Strategy
1. **Synthetic Generation**: Use GPT-4 to bootstrap initial dataset
2. **Human Curation**: Review and refine synthetic examples
3. **Augmentation**: Paraphrase and expand existing examples
4. **Diversity Metrics**: Ensure coverage of conversation types

### Quality Metrics
- **Length Distribution**: 10-200 tokens per response
- **Diversity Score**: Unique trigrams / total trigrams > 0.8
- **Safety Filtering**: Remove inappropriate content
- **Deduplication**: Fuzzy matching with threshold 0.9

### Data Pipeline
```python
# Pipeline stages
raw_data -> cleaning -> formatting -> augmentation -> validation -> training
```

Each stage has associated tests:
- `test_data_cleaning_removes_invalid_entries()`
- `test_formatting_creates_valid_json()`
- `test_augmentation_increases_diversity()`

## Performance Targets

### Inference Performance
- **Response Time**: < 2s on CPU (Intel i5+)
- **First Token Latency**: < 500ms
- **Throughput**: 5+ requests/second (batched)
- **Memory Usage**: < 4GB peak

### Training Performance
- **LoRA Fine-tuning**: < 1 hour on RTX 3060 (12GB)
- **Convergence**: Loss < 2.0 within 3 epochs
- **Gradient Accumulation**: Steps=4 for larger effective batch
- **Mixed Precision**: FP16 training for 2x speedup

### Quality Metrics
- **Perplexity**: < 20 on validation set
- **BLEU Score**: > 0.3 vs reference responses
- **Human Eval**: 80%+ "helpful" ratings
- **Safety Score**: 0% harmful outputs

## Integration Points

### Repository Integration Map
```
rasbt/LLMs-from-scratch
├── Transformer implementation reference
├── Training loop patterns
└── Evaluation metrics

huggingface/course
├── Transformers library usage
├── Dataset processing
└── Model hub integration

jaymody/picoGPT
├── Minimal implementation study
├── Core concepts validation
└── Educational reference

AK391/ai-gradio
├── Interface patterns
├── Multi-provider support
└── Deployment examples

lobehub/lobe-chat
├── UI/UX patterns
├── Conversation management
└── Plugin architecture
```

### API Design
RESTful API with WebSocket support:
```
POST /chat/completions      # OpenAI-compatible
WS   /chat/stream          # Real-time streaming
GET  /models               # Available models
POST /fine-tune            # Trigger LoRA training
```

## Technical Constraints

### Hardware Assumptions
- **Minimum**: 8GB RAM, 4-core CPU
- **Recommended**: 16GB RAM, GPU with 8GB+ VRAM
- **Storage**: 50GB for models and datasets

### Software Dependencies
- Python 3.8+ (for type hints)
- PyTorch 2.0+ (for compile() optimization)
- Transformers 4.36+ (for LoRA support)
- CUDA 11.8+ (if using GPU)

### Scaling Considerations
- Horizontal scaling via model replicas
- Quantization for edge deployment (4-bit/8-bit)
- Caching for repeated queries
- CDN for model distribution

## Development Workflow

### Feature Development Cycle
1. Research in reference repositories
2. Write behavior tests (TDD)
3. Implement minimal version
4. Validate against benchmarks
5. Optimize if needed
6. Document learnings

### Code Review Checklist
- [ ] Tests pass and cover new behavior
- [ ] Type hints added
- [ ] Docstrings updated
- [ ] Performance benchmarks met
- [ ] Memory usage acceptable
- [ ] Security considerations addressed

## Future Considerations

### Planned Enhancements
1. **Retrieval Augmented Generation (RAG)**
2. **Multi-modal support (images)**
3. **Voice interface integration**
4. **Distributed training support**
5. **Model quantization for mobile**

### Research Directions
- Mixture of Experts (MoE) for specialization
- Constitutional AI for improved safety
- Few-shot learning optimization
- Continuous learning from conversations

This context document should be updated as architectural decisions evolve and new patterns emerge from the reference repositories.