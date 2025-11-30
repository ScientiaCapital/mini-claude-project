# Execute PRP Command - TDD 6-Phase Implementation

**CRITICAL RULE**: NO OpenAI models. Use Google Gemini and ElevenLabs only.

## Usage
```bash
/execute-prp PRPs/prp_<feature>_<YYYYMMDD>.md
```

## Phase 1: Setup (15min)
```bash
git checkout -b feat/<feature>
touch tests/test_<feature>.py
touch src/<feature>.py
echo "## $(date +%Y-%m-%d): Starting <feature>" >> PLANNING.md
```

## Phase 2: Write Failing Tests (1-2hr) - TDD RED
```python
# tests/test_<feature>.py
import pytest
from src.<feature> import chatbot_response

def test_gemini_response():
    """Test Gemini generates response (NO OpenAI)."""
    response = chatbot_response("Hello")
    assert len(response) > 0
    assert "gemini" in response.lower()  # Ensure Gemini used

def test_no_openai_usage():
    """Verify NO OpenAI imports."""
    import subprocess
    result = subprocess.run(
        ["grep", "-r", "openai", "src/"],
        capture_output=True
    )
    assert result.returncode != 0, "OpenAI detected!"
```

Run: `pytest tests/test_<feature>.py` (should FAIL)

## Phase 3: Implement (2-3hr) - TDD GREEN
```python
# src/<feature>.py
import google.generativeai as genai
import os

# NO OpenAI imports
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def chatbot_response(prompt: str) -> str:
    """Generate response using Gemini (NO OpenAI)."""
    model = genai.GenerativeModel('gemini-pro')
    response = model.generate_content(prompt)
    return response.text
```

Run: `pytest tests/test_<feature>.py` (should PASS)

## Phase 4: Integration (1hr)
**Web (Next.js)**:
```typescript
// mini-claude-web/app/api/chat/route.ts
import { NextResponse } from 'next/server';

export async function POST(req: Request) {
  const { message } = await req.json();

  // Call Python backend (Gemini, not OpenAI)
  const response = await fetch('http://localhost:8000/chat', {
    method: 'POST',
    body: JSON.stringify({ message })
  });

  return NextResponse.json(await response.json());
}
```

Test: `npm test`

## Phase 5: Documentation (30min)
- Update README with TDD examples
- Document Gemini vs OpenAI rationale
- Add usage examples

## Phase 6: Final Validation (30min)
```bash
# Full test suite
npm test
pytest tests/ -v --cov

# NO OpenAI check
grep -r "openai\|gpt-" src/ mini-claude-web/ && exit 1

# Commit
git commit -m "feat(<feature>): TDD implementation

- Write failing tests first (RED)
- Implement with Gemini (GREEN)
- Add Next.js integration
- NO OpenAI models used"
```

**Success Criteria**:
- ✅ Tests pass (TDD approach)
- ✅ NO OpenAI
- ✅ Type-safe
- ✅ Coverage >80%
