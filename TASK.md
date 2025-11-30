# TASK.md - Current Work Tracking

**CRITICAL RULE**: NO OpenAI models. Use Google Gemini and ElevenLabs only.

## Active Tasks

### Current Sprint: TDD Implementation
**Status**: In Progress
**Start**: 2025-11-30
**Focus**: Test-Driven Development

#### Tasks
- [ ] Write failing tests for chatbot
- [ ] Implement Gemini integration
- [ ] Add ElevenLabs voice
- [ ] Neon database storage
- [ ] MCP server integration

---

## Today's Focus

### Priority 1: TDD Validation
```bash
npm test
pytest tests/ -v --cov
```

### Priority 2: NO OpenAI Check
```bash
grep -r "openai" src/ mini-claude-web/ && echo "❌ FAIL" || echo "✅ PASS"
```

### Priority 3: Gemini Integration
```python
import google.generativeai as genai
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
```

---

## Completed Today
- ✅ Created context engineering files
- ✅ Verified NO OpenAI usage
- ✅ TDD workflow defined

---

## Upcoming (7 Days)
- [ ] Voice chat with ElevenLabs
- [ ] Conversation history (Neon)
- [ ] Streaming responses
- [ ] MCP integration

---

## Commands

### Tests
```bash
npm test                      # Web tests
pytest tests/ -v              # Python tests
pytest tests/ --cov           # Coverage
```

### Security
```bash
grep -r "openai" . && exit 1  # NO OpenAI
```

---
**Last Updated**: 2025-11-30
**TDD Status**: RED-GREEN-REFACTOR
