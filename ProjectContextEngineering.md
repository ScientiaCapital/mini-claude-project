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

### 🔄 **Current Development Focus**
- **Voice synthesis integration**: Module complete, ready for API integration
- **Vector database tests**: pgvector and agent_memory table ready for testing
- **Vercel deployment**: Environment variables and production deployment

## Claude Code Hook System Architecture

### Agent Context Loading Pattern

The project implements a sophisticated agent context system that loads specialized knowledge based on the agent type being invoked. This enables domain-specific expertise while maintaining project coherence.

#### Agent Classification System

```typescript
interface AgentContext {
  subagent_type: 'general-purpose' | 
                 'project-docs-curator' | 
                 'devops-automation-engineer' | 
                 'bug-hunter-specialist' | 
                 'fullstack-tdd-architect' | 
                 'security-auditor-expert'
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
   - **General Purpose**: Complex multi-step tasks, code searches, research
   - **Docs Curator**: Documentation updates, learning materials, knowledge gaps
   - **DevOps Engineer**: CI/CD, deployment automation, infrastructure
   - **Bug Hunter**: Debugging, root cause analysis, issue resolution
   - **TDD Architect**: Test-driven development, architecture, code quality
   - **Security Auditor**: Security best practices, vulnerability assessment

3. **Knowledge Preservation Pipeline**:
   ```bash
   # .claude/hooks/post-agent-update.sh
   save_agent_knowledge "$AGENT_TYPE" "$TASK_OUTPUT" "$SOLUTION_PATTERNS"
   update_documentation_artifacts "$AGENT_TYPE"
   ```

### Current Implementation Context

#### Database Status
- Neon PostgreSQL with pgvector extension installed
- Tables: users, conversations, messages, agent_memory
- All connection tests passing with real database
- No mocks - using actual Neon connections

#### API Integration Status  
- Google Gemini fully integrated for chat
- ElevenLabs voice synthesis module complete
- Health monitoring endpoint functional
- Environment variables properly configured

#### Testing Status
- 95%+ test coverage maintained
- Database tests: ✅ All passing
- Voice synthesis tests: ✅ All passing
- API tests: Ready for voice integration
- Vector tests: Next priority

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
├── settings.json    # Hook configuration
├── hooks/           # Agent context scripts
│   ├── pre-task-context.sh
│   ├── post-agent-update.sh
│   └── validate-agent-work.sh
└── knowledge/       # Future: Agent learnings
```

### Benefits of Agent Specialization

1. **Contextual Expertise**: Each agent loads only relevant knowledge for their domain
2. **Knowledge Continuity**: Successful patterns are preserved across sessions
3. **Quality Assurance**: Agent-specific validation ensures domain standards
4. **Learning Amplification**: Specialized context accelerates problem-solving
5. **Documentation Automation**: Agent work automatically updates relevant documentation

## Current Technical Stack

### Production Implementation
- **Frontend**: Next.js 14 + TypeScript + Tailwind CSS
- **Backend**: Next.js API Routes with serverless functions
- **Database**: Neon PostgreSQL with pgvector extension
- **AI Models**: Google Gemini 1.5 Flash (primary)
- **Voice**: ElevenLabs API for text-to-speech
- **Deployment**: Vercel with automatic CI/CD

### Testing Approach
- **TDD First**: Every feature has tests written before implementation
- **Real Connections**: No mocks - tests use actual database and APIs
- **Coverage Target**: 95%+ maintained throughout development
- **Test Categories**: Unit, Integration, API, Database

## Learning Components (Future)

### Transformer Understanding
Educational resources in `resources/repos/`:
- **LLMs-from-scratch**: Core transformer concepts
- **picoGPT**: Minimal implementation for learning
- **huggingface/course**: Industry best practices

### Future Experiments
- **LoRA Fine-tuning**: Parameter-efficient training
- **Custom Models**: Learning transformer architecture
- **RAG Integration**: Knowledge base augmentation

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

### Educational Components
**Python MVP Chatbot**
- Located in `src/mvp_chatbot.py`
- Uses Hugging Face transformers
- For learning and experimentation

**Jupyter Notebooks**
- Interactive learning environment
- Transformer architecture exploration
- Located in `notebooks/` directory

## Database Architecture

### Current Schema
```sql
-- Core tables (implemented)
users (id, email, name, created_at, updated_at)
conversations (id, user_id, title, created_at, updated_at)
messages (id, conversation_id, role, content, created_at)

-- Vector storage (ready for testing)
agent_memory (id, agent_type, content, embedding, metadata, created_at)
```

### pgvector Integration
- Extension installed in Neon
- Vector dimension: 1536 (for OpenAI embeddings)
- Ready for similarity search implementation

## Current Performance Metrics

### API Performance (Achieved)
- **Health Check**: < 200ms response time ✅
- **Chat Response**: < 2s with Gemini API ✅
- **Database Query**: < 50ms with indexes ✅
- **Test Suite**: < 10s total runtime ✅

### Production Readiness
- **Test Coverage**: 95%+ maintained ✅
- **Error Handling**: Graceful fallbacks ✅
- **Type Safety**: Full TypeScript coverage ✅
- **Environment**: Proper secret management ✅

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

### Current API Implementation
```
POST /api/chat    # Google Gemini chat (working)
GET  /api/health  # System health check (working)
```

### Next API Features
```
POST /api/chat    # Add voice_enabled parameter
GET  /api/voices  # List available ElevenLabs voices
WS   /chat/stream # Real-time streaming responses
```

## Current Dependencies

### Production Stack
- Node.js 18+ (for Next.js)
- PostgreSQL (via Neon)
- TypeScript 5+
- React 18+

### Development Tools
- Jest for testing
- ESLint + Prettier
- Vercel CLI
- Git + GitHub

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

## Immediate Next Steps

### This Week
1. **Vector Database Tests**: Test pgvector functionality
2. **Voice Integration**: Add voice to chat API
3. **Vercel Deployment**: Deploy with environment variables
4. **Streaming Responses**: Implement real-time chat

### Future Learning
1. **Transformer Architecture**: Study and implement
2. **LoRA Fine-tuning**: Experiment with small models
3. **RAG Integration**: Add knowledge base support

This context document should be updated as architectural decisions evolve and new patterns emerge from the reference repositories.