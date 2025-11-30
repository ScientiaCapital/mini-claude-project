# Validate Command - Multi-Phase Validation

**CRITICAL RULE**: NO OpenAI models. Use Google Gemini and ElevenLabs only.

## Purpose
Run comprehensive TDD-focused validation for educational AI chatbot.

## Usage
```bash
/validate
```

## Validation Phases

### Phase 1: Environment Check
```bash
# Verify Node.js and Python
node --version  # 18+
python --version  # 3.10+

# Check .env files (API keys only in .env)
test -f mini-claude-web/.env.local && echo "✓ Web .env exists"
test -f .env && echo "✓ Python .env exists"

# Verify NO OpenAI keys
grep -i "OPENAI" .env mini-claude-web/.env.local && echo "❌ FAIL!" || echo "✅ PASS"

# Check required env vars
grep -E "GOOGLE_API_KEY|ELEVENLABS_API_KEY|NEON_DATABASE_URL" .env
```

### Phase 2: Web Tests (Next.js 14 + TypeScript)
```bash
cd mini-claude-web
npm test -- --passWithNoTests
npm run type-check  # TypeScript validation
npm run lint
```

### Phase 3: Python Tests (TDD Focus)
```bash
cd ..
pytest tests/ -v --tb=short
pytest tests/ --cov=src --cov-report=term-missing
```

### Phase 4: Database Validation (Neon PostgreSQL)
```bash
# Test database connection
python -c "
import os
from sqlalchemy import create_engine
engine = create_engine(os.getenv('NEON_DATABASE_URL'))
with engine.connect() as conn:
    print('✓ Neon PostgreSQL connected')
"
```

### Phase 5: AI Integration Tests
```bash
# Test Google Gemini (NO OpenAI)
python -c "
import google.generativeai as genai
import os
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
model = genai.GenerativeModel('gemini-pro')
response = model.generate_content('Hello')
print('✓ Gemini working')
"

# Test ElevenLabs (voice)
python -c "
from elevenlabs import generate
import os
# Test API key validity
print('✓ ElevenLabs configured')
"
```

### Phase 6: MCP Integration
```bash
# Test MCP servers if configured
python test_mcp_integration.py || echo "⚠ MCP optional"
```

## Success Criteria
- ✅ All web tests pass (npm test)
- ✅ All Python tests pass (pytest)
- ✅ NO OpenAI dependencies detected
- ✅ Google Gemini working
- ✅ ElevenLabs configured
- ✅ Neon PostgreSQL connected
- ✅ TypeScript type-safe

## Common Issues
1. **Missing dependencies**:
   - Web: `cd mini-claude-web && npm install`
   - Python: `pip install -r requirements.txt`
2. **Database connection**: Check NEON_DATABASE_URL in .env
3. **API keys**: Verify GOOGLE_API_KEY and ELEVENLABS_API_KEY

## Next Steps
- If all phases pass: Run `/generate-prp` for new features
- If failures: Fix issues and re-run `/validate`
