# mini-claude-project

**Branch**: main | **Updated**: 2025-11-30

## Status
Educational AI/ML project focused on building a functional chatbot while learning transformer architecture and LoRA fine-tuning. Currently in active development with MVP chatbot implementation using pre-trained models.

## Today's Focus
1. [ ] MVP chatbot implementation with pre-trained models
2. [ ] Transformer architecture understanding
3. [ ] LoRA fine-tuning experimentation
4. [ ] Gradio web interface development

## Done (This Session)
- (none yet)

## Critical Rules
- **NO OpenAI models** - Use DeepSeek, Qwen, Moonshot via OpenRouter
- API keys in `.env` only, never hardcoded
- Follow Test-Driven Development (TDD) principles
- Write failing tests first, then implement minimal code to pass

## Blockers
(none)

## Quick Commands
```bash
# Setup
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Run MVP chatbot
python src/mvp_chatbot.py

# Run web interface
python src/web_app.py

# Testing
pytest                          # All tests
pytest --cov=src --cov-report=html  # With coverage
pytest tests/unit/              # Unit tests only
pytest tests/integration/       # Integration tests only
ptw                            # Watch mode

# Code quality
black src/ tests/
isort src/ tests/
flake8 src/ tests/
```

## Tech Stack
- **Language**: Python 3.8+
- **Framework**: FastAPI + Uvicorn
- **AI/ML**: PyTorch, Transformers, PEFT (LoRA)
- **Testing**: pytest, pytest-cov, pytest-timeout, pytest-asyncio
- **Web**: Gradio
- **Development**: black, isort, flake8, pre-commit
- **Models**: microsoft/DialoGPT-medium
