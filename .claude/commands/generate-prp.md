# Generate PRP Command

**CRITICAL RULE**: NO OpenAI models. Use Google Gemini and ElevenLabs only.

## Usage
```bash
/generate-prp <feature-name>
```

## Process

### 1. Validate & Gather Context
```bash
/validate

# Ask user:
# - Feature description (chatbot, voice, TDD focus)
# - AI model needs (Gemini Pro/Flash)
# - Voice requirements (ElevenLabs)
# - Database schema (Neon PostgreSQL)
# - Test strategy (TDD-first)
```

### 2. Create PRP
```bash
cp PRPs/templates/prp_base.md PRPs/prp_<feature>_<YYYYMMDD>.md
```

### 3. PRP Structure
**Tech Specification**:
- **AI Model**: Google Gemini (Pro/Flash) - NO OpenAI
- **Voice**: ElevenLabs
- **Database**: Neon PostgreSQL
- **Frontend**: Next.js 14 + TypeScript
- **Backend**: Python + FastAPI

**TDD Approach**:
1. Write failing test first
2. Implement minimal code to pass
3. Refactor
4. Repeat

**Quality Gates**:
- Tests: `npm test && pytest tests/ -v`
- NO OpenAI: `grep -r "openai" . && exit 1`
- Type-safe: `npm run type-check`
- Coverage: >80%

### 4. Save PRP
```bash
git add PRPs/prp_<feature>_*.md
git commit -m "feat: PRP for <feature>"
```

## Example PRPs
- `prp_voice_chat_20251130.md` - ElevenLabs voice integration
- `prp_gemini_streaming_20251130.md` - Real-time Gemini responses
- `prp_conversation_history_20251130.md` - Neon database storage

## Next Steps
After approval: `/execute-prp <prp-file>`
