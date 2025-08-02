# Mini-Claude: Build Your Own AI Assistant

An educational project to learn transformer architecture, LoRA fine-tuning, and modern AI development through building a functional chatbot.

## Overview

Mini-Claude is a hands-on learning project that teaches you how to build an AI assistant from scratch. Following Test-Driven Development (TDD) principles, you'll implement:

- 🤖 A working chatbot using pre-trained models
- 🧠 Understanding of transformer architecture
- 🔧 LoRA (Low-Rank Adaptation) for efficient fine-tuning
- 🎨 Modern web interface with Gradio
- 📚 Integration with educational repositories

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/mini-claude.git
cd mini-claude
```

2. Create a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

1. Download required models (first time only):
```bash
# Using HuggingFace CLI (recommended)
huggingface-cli download microsoft/DialoGPT-medium

# Or use our download script
python scripts/download_models.py
```

2. Run the MVP chatbot:
```bash
python src/mvp_chatbot.py
```

3. For the web interface:
```bash
python src/web_app.py
```

## Project Structure

```
mini-claude/
├── src/                    # Core source code
│   ├── models/            # Model implementations
│   ├── training/          # Training scripts
│   └── web/               # Web interface
├── tests/                  # Test suite (TDD)
│   ├── unit/              # Unit tests
│   └── integration/       # Integration tests
├── data/                   # Training datasets
├── notebooks/              # Learning notebooks
├── resources/              # Learning materials
│   └── repos/             # Cloned repositories
└── docs/                   # Documentation
```

## Learning Path

This project follows a 12-week learning journey:

- **Weeks 1-2**: Basic chatbot and Gradio interface
- **Weeks 3-4**: Understanding transformers from scratch
- **Weeks 5-6**: Implementing LoRA fine-tuning
- **Weeks 7-8**: Training pipeline and evaluation
- **Weeks 9-10**: Advanced features (memory, streaming)
- **Weeks 11-12**: Production optimization and deployment

## Key Learning Resources

This project integrates concepts from:
- `rasbt/LLMs-from-scratch` - Core transformer education
- `huggingface/course` - Industry best practices
- `jaymody/picoGPT` - Minimal GPT implementation
- `AK391/ai-gradio` - Quick UI prototyping
- `hiyouga/LLaMA-Factory` - Advanced fine-tuning

## Testing

We follow strict TDD principles. Run tests with:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src tests/

# Run specific test category
pytest tests/unit/
```

## Contributing

This is an educational project. Feel free to:
- Report issues
- Suggest improvements
- Share your learning experience

## License

MIT License - see LICENSE file for details.

## Acknowledgments

Built as an educational project inspired by Claude and the open-source AI community.