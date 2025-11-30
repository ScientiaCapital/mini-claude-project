# PRP: <Feature Name>

**CRITICAL RULE**: NO OpenAI models. Use Google Gemini and ElevenLabs only.

## Metadata
- **PRP ID**: `prp_<feature>_<YYYYMMDD>`
- **Author**: <name>
- **Created**: <date>
- **Status**: Draft
- **TDD Approach**: Required

## Feature Overview
**Objective**: [Clear description]

**Tech Stack**:
- AI: Google Gemini (Pro/Flash)
- Voice: ElevenLabs
- Database: Neon PostgreSQL
- Frontend: Next.js 14 + TypeScript
- Backend: Python

**Success Metrics**:
- Tests pass (TDD)
- NO OpenAI usage
- Type-safe
- Coverage >80%

## Technical Specification

### AI Model (NO OpenAI)
```python
import google.generativeai as genai

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
model = genai.GenerativeModel('gemini-pro')
```

### Database Schema (Neon PostgreSQL)
```sql
CREATE TABLE conversations (
    id SERIAL PRIMARY KEY,
    user_id UUID,
    message TEXT,
    response TEXT,
    model VARCHAR(50) DEFAULT 'gemini-pro',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### Voice Integration (ElevenLabs)
```python
from elevenlabs import generate, set_api_key

set_api_key(os.getenv("ELEVENLABS_API_KEY"))
audio = generate(text="Hello", voice="Adam")
```

## TDD Implementation Phases

### Phase 1: RED (Write Failing Tests)
```python
def test_chatbot_gemini():
    response = chatbot("Hello")
    assert "gemini" in str(response.model).lower()

def test_no_openai():
    import subprocess
    result = subprocess.run(["grep", "-r", "openai", "src/"], capture_output=True)
    assert result.returncode != 0
```

### Phase 2: GREEN (Implement Minimal Code)
```python
def chatbot(prompt: str) -> dict:
    model = genai.GenerativeModel('gemini-pro')
    response = model.generate_content(prompt)
    return {"text": response.text, "model": "gemini-pro"}
```

### Phase 3: REFACTOR (Clean Code)
- Extract Gemini client
- Add error handling
- Improve type hints

### Phase 4: Integration
- Next.js API routes
- Neon database storage
- ElevenLabs voice

### Phase 5: Testing
```bash
npm test
pytest tests/ -v --cov
```

### Phase 6: Documentation
- README updates
- TDD examples
- API docs

## Quality Gates
- ✅ TDD approach (RED-GREEN-REFACTOR)
- ✅ NO OpenAI usage
- ✅ TypeScript type-safe
- ✅ Test coverage >80%
- ✅ Neon database connected

## Approval
- [ ] Tests pass
- [ ] NO OpenAI detected
- [ ] Type-safe
- [ ] Documented

---
**Remember**: NO OpenAI. TDD-first. Google Gemini + ElevenLabs only.
