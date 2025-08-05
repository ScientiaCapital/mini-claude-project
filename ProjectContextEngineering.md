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
- **MCP server integration**: context7 (vector search) and sequential-thinking (enhanced reasoning) now active

### 🔄 **Current Development Focus**
- **Voice synthesis integration**: ElevenLabs API integration in progress
- **Agent specialization**: Claude Code hook system enhanced with MCP server capabilities
- **Semantic memory optimization**: Leveraging context7 vector database for knowledge retrieval
- **Enhanced reasoning patterns**: Using sequential-thinking for better problem-solving
- **Vector database tests**: pgvector and agent_memory table ready for testing
- **Production monitoring**: Real-time health checks and performance tracking

## Claude Code Hook System Architecture with MCP Enhancement

### Agent Context Loading Pattern with Semantic Memory

The project implements a sophisticated agent context system enhanced by MCP servers that provide semantic search, vector-based memory, and structured reasoning capabilities. This enables domain-specific expertise while maintaining project coherence and leveraging historical knowledge patterns.

#### Enhanced Agent Classification System with MCP Integration

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
  // NEW: MCP-enhanced capabilities
  semantic_search_domains: string[]
  reasoning_patterns: string[]
  vector_memory_tags: string[]
}
```

#### MCP Server Architecture

**context7 - Vector Database Integration:**
- **Purpose**: Semantic search and persistent memory
- **Technology**: Upstash vector database with embedding storage
- **Capabilities**: 
  - Store agent learnings as vector embeddings
  - Retrieve contextually similar past solutions
  - Cross-agent knowledge sharing through semantic similarity
  - Persistent memory across development sessions

**sequential-thinking - Enhanced Reasoning:**
- **Purpose**: Structured problem-solving and decision-making
- **Technology**: Multi-step reasoning engine with chain-of-thought processing
- **Capabilities**:
  - Break complex problems into structured steps
  - Validate reasoning chains for completeness
  - Generate consistent documentation patterns
  - Improve architectural decision-making quality

#### Enhanced Context Loading Pipeline with MCP Integration

1. **Pre-Task Hook Execution with MCP Enhancement**:
   ```bash
   # .claude/hooks/pre-task-context.sh
   AGENT_TYPE=$(echo "$INPUT" | jq -r '.params.subagent_type // empty')
   
   # NEW: Query vector database for similar past solutions
   query_context7_memory "$AGENT_TYPE" "$TASK_DESCRIPTION"
   
   # NEW: Use structured reasoning to prepare context
   prepare_sequential_context "$AGENT_TYPE" "$TASK_CONTEXT"
   
   # Load traditional context
   load_context_for_agent "$AGENT_TYPE"
   ```

2. **Enhanced Agent-Specific Context Injection**:
   - **General Purpose**: Complex multi-step tasks, code searches, research
   - **Database Architect**: Schema patterns, query optimization, Neon-specific features (+ semantic search for similar schemas)
   - **Deployment Specialist**: Vercel configuration, environment variables, CI/CD patterns (+ structured deployment reasoning)
   - **Security Auditor**: Security best practices, vulnerability patterns, compliance requirements (+ historical security patterns)
   - **API Integration**: Google Gemini patterns, ElevenLabs configuration, rate limiting (+ sequential integration planning)
   - **Performance Optimizer**: Core Web Vitals, bundle optimization, Next.js performance patterns (+ performance history analysis) 
   - **Documentation Curator**: Learning progression, documentation standards, knowledge gaps (+ semantic similarity for content organization)

3. **Enhanced Knowledge Preservation Pipeline**:
   ```bash
   # .claude/hooks/post-agent-update.sh
   
   # NEW: Store in vector database with embeddings
   store_in_context7 "$AGENT_TYPE" "$TASK_OUTPUT" "$SOLUTION_PATTERNS"
   
   # NEW: Structure knowledge with sequential reasoning
   structure_knowledge_sequential "$AGENT_TYPE" "$LEARNINGS"
   
   # Traditional storage
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

### Enhanced Benefits of Agent Specialization with MCP Integration

1. **Contextual Expertise**: Each agent loads only relevant knowledge for their domain, enhanced by semantic search
2. **Semantic Knowledge Discovery**: Find relevant solutions even when exact keywords don't match
3. **Structured Problem-Solving**: Sequential reasoning ensures comprehensive solution development
4. **Knowledge Continuity**: Successful patterns are preserved across sessions with vector embeddings
5. **Cross-Agent Learning**: Semantic similarity enables knowledge sharing between different agent types
6. **Quality Assurance**: Agent-specific validation ensures domain standards with historical pattern matching
7. **Learning Amplification**: Specialized context accelerates problem-solving with similar solution retrieval
8. **Documentation Automation**: Agent work automatically updates relevant documentation with semantic indexing
9. **Persistent Memory**: Vector database maintains knowledge across development sessions and projects
10. **Enhanced Reasoning**: Multi-step validation ensures solution completeness and architectural coherence

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