```markdown
# Mini-Claude Project Context

**Last Updated:** 2025-10-31T13:22:49.659862

## Project Overview
- **Name:** Mini-Claude
- **Language:** Python
- **Framework:** FastAPI
- **Type:** AI/ML Educational Project

## Current Sprint Focus
**Weeks 1-2: Foundation Building**
- Basic chatbot implementation with pre-trained models
- Gradio web interface development
- Test-Driven Development (TDD) setup
- Model downloading and integration

## Architecture Overview
```
Python + FastAPI Backend
├── Transformer-based AI Models
├── Gradio Web Interface
├── LoRA Fine-tuning Capabilities
└── TDD Testing Framework
```

## Project Description
Mini-Claude is an educational AI assistant project designed to teach modern AI development through hands-on implementation. The project guides developers through building a functional chatbot from scratch while learning transformer architecture, efficient fine-tuning with LoRA, and modern web interface development. It follows Test-Driven Development principles and provides a structured 12-week learning path for comprehensive understanding of AI assistant construction.

## Recent Changes
- **Initial Project Generation** (2025-10-31)
  - Project structure setup
  - README documentation created
  - Basic directory organization established
  - Initial learning materials outlined

## Current Blockers
- None identified (project in initial setup phase)
- Potential future blockers: Model download requirements, GPU access for training

## Next Steps
1. **Setup Development Environment**
   - Complete virtual environment setup
   - Install and verify all dependencies from requirements.txt
   - Test HuggingFace CLI model downloads

2. **Implement MVP Chatbot**
   - Create basic chatbot using DialoGPT-medium model
   - Write unit tests for core functionality
   - Ensure model loading and inference works

3. **Build Web Interface**
   - Implement Gradio web application
   - Create basic chat interface
   - Test end-to-end user interaction

4. **Establish Development Workflow**
   - Set up testing pipeline
   - Create development branch structure
   - Document contribution guidelines

## Development Workflow
### Getting Started
1. Clone repository and setup virtual environment
2. Install dependencies: `pip install -r requirements.txt`
3. Download models using HuggingFace CLI or download script
4. Run tests: `pytest tests/`

### Testing Strategy
- **Unit Tests:** Core model functionality and utilities
- **Integration Tests:** End-to-end chat functionality
- **TDD Approach:** Write tests before implementation

### Branch Strategy
- `main`: Stable releases
- `develop`: Integration branch
- `feature/*`: New features and experiments
- `learning/*`: Educational implementations

## Notes
- Educational focus: Prioritize learning over production optimization
- Model choices: Start with DialoGPT-medium for rapid prototyping
- Documentation: Maintain detailed learning notes in notebooks/
- Resource management: Consider model size and computational requirements
- Community: Plan to integrate with educational AI repositories
```