# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Mini-Claude is an educational AI chatbot project that demonstrates modern AI development practices through Test-Driven Development (TDD). The project has evolved into a full-stack application featuring:

- **Production-ready web application** (Next.js + TypeScript)
- **Real-time AI chat interface** with Anthropic Claude integration
- **Scalable database architecture** (Neon PostgreSQL)
- **Complete TDD test coverage** (>95% coverage maintained)
- **Cloud deployment** ready for Vercel
- **Educational learning path** for transformer architecture and LoRA fine-tuning

## Current Project State (Completed ✅)

### Phase 1: TDD Foundation & Infrastructure
- ✅ Complete Next.js application with TypeScript
- ✅ Neon PostgreSQL database with full schema
- ✅ Anthropic Claude API integration
- ✅ Comprehensive test suite (API, Database, Components)
- ✅ Production deployment configuration
- ✅ HuggingFace CLI integration for model management

### Phase 2: Core Functionality  
- ✅ Real-time chat interface with conversation persistence
- ✅ Database-backed message history
- ✅ Error handling and validation (Zod schemas)
- ✅ Health monitoring endpoints
- ✅ Mobile-responsive design
- ✅ Security best practices (environment variables, CORS, input sanitization)

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
# Web application tests (primary)
cd mini-claude-web
npm test                          # Run all TDD tests
npm run test:coverage            # Coverage report (target: >95%)
npm test tests/api/chat.test.ts  # Specific API tests
npm test tests/database/         # Database tests

# Python component tests (educational)
pytest tests/                    # Python test suite
pytest --cov=src tests/         # Python coverage
```

### Future Training & Fine-tuning (Planned)
```bash
# LoRA fine-tuning (to be implemented)
python src/training/train_lora.py --dataset data/conversations/mini_claude_v1.json --epochs 3

# Model evaluation (to be implemented)
python src/evaluation/evaluate.py --model models/mini_claude_lora --test-set data/test_conversations.json
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

4. **AI Integration**: Anthropic Claude API
   - Model: claude-3-sonnet-20240229
   - Conversation context maintained
   - Error handling and rate limiting
   - Response streaming support

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
   - Voice synthesis integration (ElevenLabs)
   - Multi-modal support (images, documents)
   - RAG integration for knowledge base
   - Real-time streaming responses

## Key Technical Decisions

### Current Architecture Decisions
1. **Next.js over Python**: Better for production deployment and scalability
2. **Anthropic Claude over Local Models**: Higher quality responses, lower infrastructure costs
3. **Neon PostgreSQL**: Serverless, auto-scaling database with excellent Vercel integration
4. **TypeScript**: Type safety for production reliability
5. **TDD Philosophy**: Every feature implemented tests-first for reliability

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

Follow the detailed learning progression in **ProjectTasks.md** for educational components.