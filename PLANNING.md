# PLANNING.md - Mini Claude Project

**CRITICAL RULE**: NO OpenAI models. Use Google Gemini and ElevenLabs only.

## Project Overview
Educational AI chatbot with TDD focus.

**Tech Stack**:
- Next.js 14 + TypeScript
- Python + FastAPI
- Google Gemini (NO OpenAI)
- ElevenLabs (voice)
- Neon PostgreSQL
- MCP integration

## Architecture Decisions

### ADR-001: NO OpenAI Models
**Date**: 2025-11-30
**Status**: **MANDATORY**

**Decision**: Use Google Gemini only

**Enforcement**:
```bash
grep -r "openai\|gpt-" src/ mini-claude-web/ && exit 1
```

### ADR-002: TDD Approach
**Date**: 2025-11-30
**Status**: Required

**Process**: RED-GREEN-REFACTOR
1. Write failing test
2. Implement minimal code
3. Refactor

### ADR-003: Google Gemini
**Date**: 2025-11-30
**Status**: Adopted

**Models**:
- Gemini Pro: Complex reasoning
- Gemini Flash: Fast responses

**Cost**: ~$0.0001/call (cheap)

### ADR-004: ElevenLabs Voice
**Date**: 2025-11-30
**Status**: Adopted

**Use**: Text-to-speech for chatbot
**Cost**: ~$0.30/1K characters

### ADR-005: Neon PostgreSQL
**Date**: 2025-11-30
**Status**: Adopted

**Schema**: Conversations, users
**Cost**: Free tier

## Module Structure
```
mini-claude-project/
├── mini-claude-web/          # Next.js 14
│   ├── app/
│   │   └── api/chat/route.ts
│   └── tests/
├── src/                      # Python backend
│   ├── chatbot.py
│   └── voice.py
├── tests/                    # Python tests (TDD)
│   ├── test_chatbot.py
│   └── test_voice.py
├── PRPs/
├── PLANNING.md               # This file
└── TASK.md
```

## TDD Workflow
1. Write test: `def test_feature()`
2. Run: `pytest` (FAIL - RED)
3. Implement: Minimal code
4. Run: `pytest` (PASS - GREEN)
5. Refactor: Clean code
6. Commit

## Dependencies
**Python**:
```
google-generativeai>=0.3.0
elevenlabs>=0.2.0
fastapi>=0.104.0
sqlalchemy>=2.0.0
```

**PROHIBITED**:
```
openai              # ❌ NEVER
```

**Web**:
```json
{
  "@google/generative-ai": "^0.1.0",
  "next": "14.0.0",
  "typescript": "^5.0.0"
}
```

## Performance Targets
| Metric | Target |
|--------|--------|
| Response time | <2s |
| Voice generation | <1s |
| Test coverage | >80% |

---
**Last Updated**: 2025-11-30
