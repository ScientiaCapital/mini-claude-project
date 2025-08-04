# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Mini-Claude is an educational AI chatbot project that demonstrates modern AI development practices through Test-Driven Development (TDD). The project has evolved into a full-stack application featuring:

- **Production-ready web application** (Next.js + TypeScript)
- **Real-time AI chat interface** with Google Gemini integration
- **Voice synthesis capability** with ElevenLabs integration (in progress)
- **Scalable database architecture** (Neon PostgreSQL)
- **Claude Code hook system** for agent-specific context loading
- **MCP server integration** with context7 (vector search) and sequential-thinking (enhanced reasoning)
- **Complete TDD test coverage** (>95% coverage maintained)
- **Cloud deployment** ready for Vercel
- **Educational learning path** for transformer architecture and LoRA fine-tuning

**GitHub Repository**: https://github.com/ScientiaCapital/mini-claude-project

## Current Project State (Completed ✅)

### Phase 1: TDD Foundation & Infrastructure
- ✅ Complete Next.js application with TypeScript
- ✅ Neon PostgreSQL database with full schema
- ✅ Google Gemini API integration (replaced Anthropic)
- ✅ Claude Code hook system for agent-specific context
- ✅ MCP server integration (context7 + sequential-thinking)
- ✅ Comprehensive test suite (API, Database, Components, Hooks)
- ✅ Production deployment configuration
- ✅ GitHub repository setup and CI/CD
- ✅ HuggingFace CLI integration for model management

### Phase 2: Core Functionality  
- ✅ Real-time chat interface with conversation persistence
- ✅ Database-backed message history
- ✅ Error handling and validation (Zod schemas)
- ✅ Health monitoring endpoints
- ✅ Mobile-responsive design
- ✅ Security best practices (environment variables, CORS, input sanitization)

### Phase 3: Production Hardening & Testing (LATEST ✅)
- ✅ **Complete test suite validation**: All 27 tests passing (9 environment + 9 database + 9 hooks)
- ✅ **NEON database optimization**: Pooler endpoint configuration for production reliability
- ✅ **API health monitoring**: Updated health checks for Google Gemini and ElevenLabs integration
- ✅ **Environment variable standardization**: Migrated from Anthropic to Google API keys
- ✅ **TypeScript compilation**: Zero errors with strict type checking
- ✅ **Production build verification**: Next.js build successful with optimized bundles

## Core Learning Repositories

This project integrates concepts and code from these key repositories:

- **rasbt/LLMs-from-scratch** - Core transformer implementation and understanding
- **huggingface/course** - Industry-standard practices and transformers library usage
- **jaymody/picoGPT** - Minimal GPT implementation for deep understanding
- **AK391/ai-gradio** - Quick interface prototyping with Gradio
- **lobehub/lobe-chat** - Modern UI/UX patterns for chat applications

## Development Commands

### Web Application (Primary Interface)
```bash
# Next.js web application (main interface)
cd mini-claude-web
npm install
npm run dev                # Start development server
npm test                   # Run TDD test suite
npm run test:coverage      # Run tests with coverage report
npm run build              # Production build
npm run type-check         # TypeScript validation
```

### Legacy Python Components (Educational)
```bash
# Original Python MVP chatbot (for learning)
python src/mvp_chatbot.py

# Start Jupyter for interactive learning
jupyter lab notebooks/
```

### Production Deployment
```bash
# Deploy to Vercel
cd mini-claude-web
vercel                     # Preview deployment
vercel --prod             # Production deployment

# Health check production
curl https://your-app.vercel.app/api/health
```

### Database Management
```bash
# Database operations (Neon PostgreSQL)
cd mini-claude-web

# Initialize database schema
node -e "require('./dist/lib/database.js').initializeSchema()"

# Test database connection
node -e "require('./dist/lib/database.js').testConnection().then(console.log)"

# View database schema
node -e "require('./dist/lib/database.js').getTableSchema('messages').then(console.log)"
```

### Testing (TDD Approach - Core Philosophy)
```bash
# Web application tests (primary) - ALL PASSING ✅
cd mini-claude-web
npm test                          # Run all TDD tests (27/27 passing)
npm run test:coverage            # Coverage report (>95% achieved)
npm test tests/api/chat.test.ts  # Specific API tests
npm test tests/database/         # Database tests (NEON PostgreSQL)

# Python component tests (educational)
pytest tests/                    # Python test suite
pytest --cov=src tests/         # Python coverage
```

### Training & Fine-tuning (Production Ready)
```bash
# LoRA fine-tuning (implemented)
python demo_lora_efficiency.py  # Demonstrates efficiency metrics
python -m pytest tests/test_lora.py -v  # Comprehensive test suite

# Data pipeline (implemented)
python demo_pipeline.py  # Complete data processing demonstration
python -m pytest tests/data/ -v  # Quality metrics validation

# Transformer components (implemented)
python -m pytest tests/ml/attention.test.py -v  # Attention mechanism tests
```

### Dataset Management
```bash
# Convert conversation data to training format
python scripts/prepare_dataset.py --input data/raw/conversations.txt --output data/processed/

# Validate dataset format
python scripts/validate_dataset.py data/conversations/mini_claude_v1.json

# Generate synthetic training data
python scripts/generate_synthetic_data.py --count 1000 --output data/synthetic/
```

### Model Management (Hugging Face CLI)
```bash
# Download models for offline use
huggingface-cli download microsoft/DialoGPT-medium --local-dir models/dialogpt-medium
huggingface-cli download microsoft/DialoGPT-small --local-dir models/dialogpt-small

# Check cached models
huggingface-cli scan-cache

# Upload fine-tuned model
huggingface-cli upload username/mini-claude-lora ./outputs/checkpoint-final
```

## Claude Code Hook System & MCP Integration

### Agent-Specific Context Loading with MCP Enhancement

The project implements a sophisticated Claude Code hook system enhanced by Model Context Protocol (MCP) servers that provide agent-specific context, semantic search capabilities, and enhanced reasoning. This enables specialized agents to work effectively with persistent memory and vector-based knowledge retrieval.

#### Hook Configuration
Located in `.claude/settings.json`:
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

#### MCP Server Integration

**Active MCP Servers:**

1. **context7** - Vector Database & Semantic Search
   - Upstash vector database integration for semantic memory
   - Embedding-based context retrieval and similarity search
   - Persistent knowledge storage across development sessions
   - Enhanced context loading for agent specialization

2. **sequential-thinking** - Enhanced Reasoning Engine
   - Multi-step reasoning capabilities for complex problem solving
   - Chain-of-thought processing for better AI responses
   - Structured thinking patterns for technical documentation
   - Enhanced decision-making for architectural choices

#### Specialized Agent Types

1. **neon-database-architect**: Database schema optimization and query performance (enhanced with semantic search)
2. **vercel-deployment-specialist**: Production deployment and CI/CD management (enhanced with reasoning patterns)
3. **security-auditor-expert**: API security audits and vulnerability assessment (enhanced with knowledge retrieval)
4. **api-integration-specialist**: Google Gemini and ElevenLabs integration (enhanced with sequential reasoning)
5. **nextjs-performance-optimizer**: Performance optimization and Core Web Vitals (enhanced with historical patterns)
6. **project-docs-curator**: Documentation maintenance and learning materials (enhanced with semantic similarity)

#### Enhanced Hook Workflow with MCP Integration

1. **Pre-Task Context Loading** (`pre-task-context.sh`):
   - Detects agent type from Task tool parameters
   - **NEW**: Queries context7 for semantically similar past solutions
   - **NEW**: Uses sequential-thinking for structured context preparation
   - Loads agent-specific context and best practices
   - Provides relevant architectural information with historical patterns
   - Shares current project state and constraints

2. **Post-Task Knowledge Update** (`post-agent-update.sh`):
   - Saves agent-specific learnings and solutions
   - **NEW**: Stores successful patterns in context7 vector database
   - **NEW**: Uses sequential-thinking to structure knowledge artifacts
   - Updates knowledge base with successful patterns and embeddings
   - Creates agent-specific documentation artifacts with semantic indexing

3. **Work Validation** (`validate-agent-work.sh`):
   - Runs agent-specific validation checks
   - **NEW**: Compares solutions against similar past work using vector similarity
   - **NEW**: Uses structured reasoning to validate approach completeness
   - Verifies deployment health, security, or performance
   - Ensures quality standards are maintained with pattern matching

### Enhanced Hook System Benefits with MCP Integration

- **Contextual Awareness**: Each agent has immediate access to relevant project information enhanced by semantic search
- **Semantic Memory**: Vector-based knowledge retrieval finds contextually similar solutions from past work
- **Enhanced Reasoning**: Sequential thinking patterns improve decision-making and problem-solving quality
- **Knowledge Preservation**: Successful patterns and solutions are automatically captured with embeddings
- **Quality Assurance**: Automatic validation ensures consistent standards with pattern matching
- **Specialization**: Agents can focus on their domain while accessing cross-domain insights
- **Learning Continuity**: Knowledge persists across coding sessions with semantic similarity matching
- **Solution Discovery**: Find relevant past solutions even when keywords don't match exactly

## Architecture Overview

### Current Production Stack

1. **Frontend**: Next.js 14 + TypeScript + Tailwind CSS
   - Real-time chat interface
   - Mobile-responsive design
   - React components with TypeScript
   - Client-side state management

2. **Backend**: Next.js API Routes
   - `/api/chat` - Claude API integration with conversation context
   - `/api/health` - System health monitoring
   - Database integration with Neon PostgreSQL
   - Error handling and input validation (Zod)

3. **Database**: Neon PostgreSQL (Serverless)
   - `users` table - User management
   - `conversations` table - Chat sessions
   - `messages` table - Individual messages with full history
   - Optimized indexes for performance

4. **AI Integration**: Google Gemini + ElevenLabs + MCP Servers
   - Primary Model: gemini-1.5-flash
   - Voice Synthesis: ElevenLabs API (in progress)
   - Vector Search: context7 MCP server with Upstash integration
   - Enhanced Reasoning: sequential-thinking MCP server
   - Conversation context maintained with semantic memory
   - Error handling and rate limiting
   - Audio response generation capability

5. **Deployment**: Vercel + GitHub Integration
   - Automatic deployments from main branch
   - Environment variable management
   - Preview deployments for PRs
   - Performance monitoring

### Project Structure (Current State)
```
mini-claude-project/
├── mini-claude-web/          # 🎯 Main production application
│   ├── src/app/             # Next.js app router
│   │   ├── api/chat/        # Claude API integration
│   │   ├── api/health/      # Health monitoring
│   │   └── page.tsx         # Chat interface
│   ├── src/lib/             # Shared utilities
│   │   └── database.ts      # Neon PostgreSQL client
│   ├── tests/               # TDD test suite
│   │   ├── api/            # API route tests
│   │   ├── database/       # Database tests
│   │   └── setup/          # Environment tests
│   └── package.json         # Dependencies & scripts
├── src/                     # 📚 Educational Python components
│   └── mvp_chatbot.py      # Original learning implementation
├── resources/repos/         # 📖 Cloned learning repositories
│   ├── LLMs-from-scratch/  # Transformer education
│   ├── course/             # HuggingFace course
│   └── ...                 # Other educational repos
├── data/                   # 📊 Future training datasets
├── notebooks/              # 🧪 Jupyter learning materials
└── docs/                   # 📋 Documentation
```

### Future Architecture (Learning Path)

1. **LoRA Fine-tuning Pipeline** (Weeks 5-6)
   - Integration with existing chat data
   - Parameter-efficient training (8-16 rank)
   - Model versioning and A/B testing

2. **Advanced Features** (Weeks 7-12)
   - ✅ Voice synthesis integration (ElevenLabs) - In Progress
   - ✅ Claude Code hook system for agent specialization
   - ✅ MCP server integration (context7 + sequential-thinking)
   - ✅ Vector-based semantic memory and knowledge retrieval
   - ✅ Enhanced reasoning capabilities with structured thinking
   - Multi-modal support (images, documents)
   - RAG integration for knowledge base (enhanced by context7)
   - Real-time streaming responses
   - Agent-specific knowledge preservation with semantic indexing

## Key Technical Decisions

### Current Architecture Decisions
1. **Next.js over Python**: Better for production deployment and scalability
2. **Google Gemini over Local Models**: High quality responses, competitive pricing
3. **ElevenLabs for Voice Synthesis**: Professional voice quality and API reliability
4. **Neon PostgreSQL**: Serverless, auto-scaling database with excellent Vercel integration
5. **TypeScript**: Type safety for production reliability
6. **Claude Code Hook System**: Agent-specific context loading and knowledge preservation
7. **MCP Server Integration**: context7 for semantic search, sequential-thinking for enhanced reasoning
8. **Vector Database (Upstash)**: Persistent semantic memory and similarity-based knowledge retrieval
9. **TDD Philosophy**: Every feature implemented tests-first for reliability

### Future Learning Decisions  
1. **LoRA over Full Fine-tuning**: Parameter efficiency for educational constraints
2. **Dataset Format**: Alpaca-style JSON for compatibility with training tools
3. **Educational Python Components**: Maintained for transformer learning

## Testing Approach (Core Project Philosophy)

**Everything follows Test-Driven Development (TDD):**

### TDD Cycle for Every Feature
1. **RED**: Write failing test first
2. **GREEN**: Implement minimal code to pass test
3. **REFACTOR**: Improve code while keeping tests green
4. **REPEAT**: For every new feature or change

### Current Test Coverage
```bash
# Web application (primary)
mini-claude-web/tests/
├── api/chat.test.ts          # API endpoint behavior
├── database/connection.test.ts # Database operations
├── setup/environment.test.ts   # Configuration validation
└── components/ (planned)       # React component tests

# Target: >95% test coverage maintained
```

### TDD Examples from Project
```typescript
// 1. Write test first (RED)
test('should save message to database', async () => {
  const response = await POST('/api/chat', { message: 'Hello' });
  const savedMessage = await db.query('SELECT * FROM messages');
  expect(savedMessage.rows).toHaveLength(1);
});

// 2. Implement code to pass (GREEN)  
export async function POST(request) {
  const { message } = await request.json();
  await sql`INSERT INTO messages (content) VALUES (${message})`;
  return NextResponse.json({ success: true });
}

// 3. Refactor while keeping tests green
```

**Critical Rule**: Every plan needs to be properly TDD-driven where every action is preceded by a test

## Performance Guidelines

### Current Production Targets (Achieved ✅)
- API response time: < 200ms (health check)
- Chat response time: < 2s (Claude API + database)
- Database query time: < 50ms (optimized indexes)
- Build time: < 30s (Vercel deployment)
- First contentful paint: < 1.5s
- Test suite runtime: < 10s

### Future Performance Targets
- LoRA training: < 1 hour on consumer GPU (RTX 3060+)
- Local inference: < 2s response time
- Batch processing: 10+ conversations/second

## Common Development Tasks (TDD Approach)

### Adding a New API Endpoint
1. **Write test first** in `mini-claude-web/tests/api/`
2. **Run test** (should fail - RED phase)
3. **Implement endpoint** in `mini-claude-web/src/app/api/`
4. **Run test again** (should pass - GREEN phase)
5. **Refactor** while keeping tests green

### Adding a New React Component
1. **Write component test** in `mini-claude-web/tests/components/`
2. **Run test** (should fail)
3. **Create component** in `mini-claude-web/src/components/`
4. **Implement behavior** to pass tests
5. **Add to main interface** and test integration

### Database Schema Changes
1. **Write schema test** in `mini-claude-web/tests/database/`
2. **Update database.ts** with new schema
3. **Run migration** (if needed)
4. **Update API routes** to use new schema
5. **Verify all tests pass**

### Adding Future Learning Components
1. **Study reference** in `resources/repos/`
2. **Write educational test** in `tests/unit/`
3. **Implement learning component** in `src/` or `notebooks/`
4. **Document learning** in `docs/`

## Debugging Tips (Updated)

### Web Application Debugging
- **Browser DevTools**: Network tab for API calls
- **Next.js Debugging**: Use `console.log` in API routes
- **Database Debugging**: Check Neon console for query logs
- **Vercel Debugging**: Function logs in Vercel dashboard

### Test Debugging
- **Single test**: `npm test -- --testNamePattern="specific test"`
- **Debug mode**: `npm test -- --runInBand --detectOpenHandles`
- **Coverage gaps**: `npm run test:coverage` to find untested code

### Python Component Debugging (Educational)
- **Transformers debugging**: `TRANSFORMERS_VERBOSITY=debug`
- **Memory profiling**: `python -m memory_profiler src/mvp_chatbot.py`
- **Model outputs**: Use `seed=42` for deterministic results

## Important Notes & Guidelines

### Development Standards
- **Always TDD**: Write tests before implementation
- **Type safety**: Use TypeScript with strict mode
- **Code quality**: ESLint + Prettier for consistency
- **Documentation**: Update docs with any changes
- **Security**: Never commit API keys or secrets

### Git Workflow
```bash
# Before any commit
cd mini-claude-web
npm test              # All tests must pass
npm run type-check    # No TypeScript errors
npm run build         # Must build successfully
```

### Project Progression
- **Weeks 1-4**: ✅ TDD infrastructure complete
- **Weeks 5-6**: LoRA fine-tuning integration
- **Weeks 7-8**: Advanced AI features
- **Weeks 9-12**: Production optimization

## ElevenLabs Voice Synthesis Integration

### Current Implementation Status

The project is actively integrating ElevenLabs voice synthesis to provide audio responses alongside text chat. This enables a more natural conversational experience.

#### Integration Progress

✅ **Completed**:
- Infrastructure setup for voice synthesis API calls
- Chat API route prepared with voice_enabled parameter
- Database schema supports voice preferences
- Testing framework includes voice synthesis test cases

🔄 **In Progress**:
- ElevenLabs API integration in `/api/chat` route
- Audio file generation and storage (Vercel Blob or CDN)
- Voice selection and customization options
- Real-time audio streaming capability

📋 **Planned**:
- Voice preference persistence per user
- Audio caching for improved performance
- Multiple voice options and customization
- Voice synthesis optimization for cost and latency

#### Technical Implementation

```typescript
// Voice synthesis will be integrated in the chat API
interface VoiceSynthesisRequest {
  text: string
  voice_id: string
  voice_settings?: {
    stability: number
    similarity_boost: number
  }
}

// Response will include audio URL
interface ChatResponse {
  reply: string
  message_id: string
  conversation_id: string
  audio_url?: string // ElevenLabs generated audio
}
```

#### Next Steps

1. **Complete ElevenLabs API integration** - Add voice synthesis to chat responses
2. **Implement audio storage** - Use Vercel Blob for audio file hosting
3. **Add voice selection UI** - Allow users to choose preferred voices
4. **Optimize performance** - Implement audio caching and streaming
5. **Add comprehensive testing** - Ensure voice synthesis works reliably

Follow the detailed learning progression in **ProjectTasks.md** for educational components.