# ProjectContextEngineering.md

## Technical Context and Engineering Decisions for Mini-Claude

This document captures the technical context, architectural decisions, and engineering rationale for the Mini-Claude project. Updated to reflect the current production state and future learning objectives.

## Current Implementation Status (Updated)

### ✅ **Production Architecture Completed**
**Stack**: Next.js 14 + TypeScript + Neon PostgreSQL + Google Gemini + ElevenLabs + Vercel

**Key Engineering Decisions Made:**
1. **Next.js over Python Flask/FastAPI**: Better TypeScript integration, Vercel deployment
2. **Google Gemini over Anthropic/local models**: Competitive pricing, high quality responses, Google ecosystem integration
3. **ElevenLabs for voice synthesis**: Professional voice quality, extensive voice library, reliable API
4. **Neon PostgreSQL over SQLite**: Serverless scaling, better production reliability
5. **Claude Code hook system**: Agent-specific context loading and knowledge preservation
6. **TDD-first development**: 95%+ test coverage maintained throughout

### ✅ **Recent Technical Achievements (Latest)**
- **Complete test suite validation**: 27/27 tests passing with comprehensive coverage
- **NEON database production optimization**: Pooler endpoint implementation for scalability
- **API health monitoring standardization**: Google Gemini and ElevenLabs integration validation
- **Environment variable security**: Complete migration from development to production keys
- **TypeScript strict mode compliance**: Zero compilation errors with enhanced type safety

### 🔄 **Current Development Focus**
- **Voice synthesis integration**: ElevenLabs API integration in progress
- **Agent specialization**: Claude Code hook system enables domain-specific agents
- **Production monitoring**: Real-time health checks and performance tracking
- **Educational ML components**: Transformer implementation and LoRA fine-tuning ready for deep learning exploration

## Claude Code Hook System Architecture

### Agent Context Loading Pattern

The project implements a sophisticated agent context system that loads specialized knowledge based on the agent type being invoked. This enables domain-specific expertise while maintaining project coherence.

#### Agent Classification System

```typescript
interface AgentContext {
  subagent_type: 'neon-database-architect' | 
                 'vercel-deployment-specialist' | 
                 'security-auditor-expert' | 
                 'api-integration-specialist' | 
                 'nextjs-performance-optimizer' | 
                 'project-docs-curator'
  required_context: string[]
  memory_focus: string[]
  documentation_responsibilities: string[]
}
```

#### Context Loading Pipeline

1. **Pre-Task Hook Execution**:
   ```bash
   # .claude/hooks/pre-task-context.sh
   AGENT_TYPE=$(echo "$INPUT" | jq -r '.params.subagent_type // empty')
   load_context_for_agent "$AGENT_TYPE"
   ```

2. **Agent-Specific Context Injection**:
   - **Database Architect**: Schema patterns, query optimization, Neon-specific features
   - **Deployment Specialist**: Vercel configuration, environment variables, CI/CD patterns
   - **Security Auditor**: Security best practices, vulnerability patterns, compliance requirements
   - **API Integration**: Google Gemini patterns, ElevenLabs configuration, rate limiting
   - **Performance Optimizer**: Core Web Vitals, bundle optimization, Next.js performance patterns
   - **Documentation Curator**: Learning progression, documentation standards, knowledge gaps

3. **Knowledge Preservation Pipeline**:
   ```bash
   # .claude/hooks/post-agent-update.sh
   save_agent_knowledge "$AGENT_TYPE" "$TASK_OUTPUT" "$SOLUTION_PATTERNS"
   update_documentation_artifacts "$AGENT_TYPE"
   ```

### Agent-Specific Context Requirements

#### Database Architect Context
- Current Neon PostgreSQL schema and relationships
- Query performance optimization patterns
- Connection pooling and serverless database best practices
- Data migration and versioning strategies

#### Deployment Specialist Context  
- Vercel deployment configuration and environment management
- CI/CD pipeline status and optimization opportunities
- Performance monitoring and alerting setup
- Deployment rollback and recovery procedures

#### Security Auditor Context
- API key management and rotation policies
- OWASP security checklist compliance
- Vulnerability assessment results and remediation
- Data privacy and compliance requirements

#### API Integration Specialist Context
- Google Gemini API configuration and best practices
- ElevenLabs voice synthesis integration patterns
- Rate limiting and error handling strategies
- API cost optimization and usage monitoring

#### Performance Optimizer Context
- Current Core Web Vitals scores and trends
- Bundle analysis and code splitting opportunities
- Image optimization and CDN configuration
- Runtime performance profiling results

#### Documentation Curator Context
- Learning milestone progress and gaps
- Documentation quality standards and templates
- User feedback and knowledge transfer requirements
- Educational content effectiveness metrics

### Hook System Implementation

#### Settings Configuration
```json
{
  "hooks": {
    "PreToolUse": [
      {
        "matcher": "Task",
        "hooks": [
          {
            "type": "command",
            "command": ".claude/hooks/pre-task-context.sh"
          }
        ]
      }
    ],
    "PostToolUse": [
      {
        "matcher": "Task", 
        "hooks": [
          {
            "type": "command",
            "command": ".claude/hooks/post-agent-update.sh"
          },
          {
            "type": "command", 
            "command": ".claude/hooks/validate-agent-work.sh"
          }
        ]
      }
    ]
  }
}
```

#### Knowledge Persistence Structure
```
.claude/
├── knowledge/
│   ├── agents/
│   │   ├── neon-database-architect/
│   │   │   ├── successful-patterns.md
│   │   │   ├── optimization-history.json
│   │   │   └── troubleshooting-guide.md
│   │   ├── vercel-deployment-specialist/
│   │   │   ├── deployment-patterns.md
│   │   │   ├── environment-config.json
│   │   │   └── performance-metrics.json
│   │   └── ...
│   └── shared/
│       ├── architecture-decisions.md
│       ├── tdd-patterns.md
│       └── integration-patterns.md
```

### Benefits of Agent Specialization

1. **Contextual Expertise**: Each agent loads only relevant knowledge for their domain
2. **Knowledge Continuity**: Successful patterns are preserved across sessions
3. **Quality Assurance**: Agent-specific validation ensures domain standards
4. **Learning Amplification**: Specialized context accelerates problem-solving
5. **Documentation Automation**: Agent work automatically updates relevant documentation

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

### Production Model: Google Gemini 1.5 Flash
**Why Google Gemini?**
- High-quality responses with competitive pricing
- Excellent conversation context handling
- Fast response times suitable for real-time chat
- Strong instruction following capabilities
- Google ecosystem integration benefits

### Voice Synthesis: ElevenLabs
**Why ElevenLabs?**
- Professional voice quality with natural speech patterns
- Extensive voice library with customization options
- Reliable API with good documentation
- Real-time voice synthesis capabilities
- Competitive pricing for production use

### MVP Model: DialoGPT-medium (Educational)
**Why DialoGPT-medium for learning?**
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
RESTful API with voice synthesis support:
```
POST /api/chat             # Google Gemini chat with optional voice synthesis
GET  /api/health           # System health monitoring
POST /api/voice/synthesize # ElevenLabs voice generation (planned)
GET  /api/voice/voices     # Available voice options (planned)
WS   /chat/stream          # Real-time streaming (planned)
POST /fine-tune            # LoRA training (educational)
```

#### Current API Implementation
- **Primary Chat Endpoint**: `/api/chat` - Google Gemini integration with conversation persistence
- **Voice Synthesis**: ElevenLabs integration in progress for audio response generation
- **Health Monitoring**: Production-ready health checks for deployment monitoring
- **Database Integration**: Neon PostgreSQL for conversation history and user management

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

## Database Engineering Decisions

### NEON PostgreSQL Production Configuration

**Connection Architecture:**
- **Pooler Endpoint**: `ep-blue-frog-aex25r9z-pooler.c-2.us-east-2.aws.neon.tech`
- **Connection Parameters**: `channel_binding=require&sslmode=require`
- **Database**: `neondb` with role `neondb_owner`

**Schema Design:**
```sql
-- Users table
CREATE TABLE users (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  email VARCHAR(255) UNIQUE NOT NULL,
  name VARCHAR(255),
  created_at TIMESTAMPTZ DEFAULT NOW(),
  updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Conversations table  
CREATE TABLE conversations (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID REFERENCES users(id),
  title TEXT DEFAULT 'New Conversation',
  created_at TIMESTAMPTZ DEFAULT NOW(),
  updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Messages table
CREATE TABLE messages (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  conversation_id UUID REFERENCES conversations(id) ON DELETE CASCADE,
  role TEXT NOT NULL CHECK (role IN ('user', 'assistant', 'system')),
  content TEXT NOT NULL,
  created_at TIMESTAMPTZ DEFAULT NOW()
);
```

**Performance Optimizations:**
- Connection pooling for serverless scalability
- Indexed foreign keys for efficient joins
- Cascading deletes for data consistency
- UUID primary keys for distributed architecture
- Timestamp tracking for all entities

**Testing Strategy:**
- Environment-specific test configurations
- Database schema validation tests
- Connection reliability testing
- Performance benchmarking under load

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