# Repository Guide for Mini-Claude Project

This guide documents the educational repositories we use and how they contribute to our learning journey.

## Core Educational Repositories

### 1. LLMs-from-scratch
**Path**: `resources/repos/LLMs-from-scratch`  
**Purpose**: Core transformer implementation and understanding

**Key Learning Chapters**:
- **Chapter 2**: Working with text data, tokenization
- **Chapter 3**: Attention mechanisms (crucial for understanding transformers)
- **Chapter 4**: Implementing GPT from scratch
- **Chapter 5**: Pretraining on unlabeled data
- **Chapter 6**: Fine-tuning for classification
- **Chapter 7**: Instruction fine-tuning

**How we use it**: Reference implementation for our transformer components in `src/models/transformer_components.py`

### 2. course
**Path**: `resources/repos/course`  
**Purpose**: Industry-standard practices using Hugging Face transformers

**Key Sections**:
- `chapters/en/chapter1/`: Transformer models overview
- `chapters/en/chapter2/`: Using Hugging Face Transformers
- `chapters/en/chapter3/`: Fine-tuning a pretrained model
- `chapters/en/chapter7/`: Main NLP tasks

**How we use it**: Best practices for using transformers library in production

### 3. ai-gradio
**Path**: `resources/repos/ai-gradio`  
**Purpose**: Quick interface building for our chatbot

**Key Features**:
- Multi-provider AI integration
- Simple one-line interfaces
- Voice and video chat examples

**How we use it**: Template for our Gradio interfaces in `src/web/`

### 4. LLaMA-Factory
**Path**: `resources/repos/LLaMA-Factory`  
**Purpose**: Advanced training and fine-tuning framework

**Key Components**:
- `src/llmtuner/`: Core training implementation
- Web UI for no-code fine-tuning
- Support for LoRA, QLoRA, and full fine-tuning

**How we use it**: Reference for implementing our LoRA training in `src/training/train_lora.py`

### 5. lobe-chat
**Path**: `resources/repos/lobe-chat`  
**Purpose**: Modern UI/UX patterns for chat applications

**Key Patterns**:
- Conversation management
- Streaming responses
- Multi-modal interactions
- Plugin architecture

**How we use it**: UI/UX inspiration for advanced features

## Additional Learning Repositories

### 6. picoGPT
**Path**: `resources/repos/picoGPT`  
**Purpose**: Minimal GPT implementation in 40 lines

**Key File**: `gpt2.py` - Shows the core math behind transformers

**How we use it**: Understanding the essential components without abstraction

### 7. LoRA
**Path**: `resources/repos/LoRA`  
**Purpose**: Official Microsoft LoRA implementation

**Key Concepts**:
- Low-rank matrix decomposition
- Parameter-efficient fine-tuning
- Integration with existing models

**How we use it**: Reference for our LoRA implementation in `src/models/lora.py`

## Learning Path Integration

### Week 1-2: Foundation
- Start with `picoGPT` for minimal understanding
- Use `ai-gradio` for quick UI

### Week 3-4: Deep Dive
- Study `LLMs-from-scratch` Ch. 2-3
- Follow `course` Ch. 1-2

### Week 5-6: LoRA Implementation
- Reference `LoRA` repository
- Use `LLaMA-Factory` for practical examples

### Week 7-8: Training
- `LLMs-from-scratch` Ch. 5-6
- `course` Ch. 3 on fine-tuning

### Week 9-12: Advanced Features
- `lobe-chat` for UI patterns
- `LLaMA-Factory` for production features

## Quick Reference Commands

```bash
# View picoGPT implementation
cat resources/repos/picoGPT/gpt2.py

# Open LLMs-from-scratch notebook
jupyter lab resources/repos/LLMs-from-scratch/ch02/01_main-chapter-code/ch02.ipynb

# Browse course materials
open resources/repos/course/chapters/en/

# Check LLaMA-Factory training scripts
ls resources/repos/LLaMA-Factory/src/llmtuner/train/
```

## Tips for Using These Resources

1. **Don't just copy code** - Understand the concepts first
2. **Run the examples** - Each repo has working code to experiment with
3. **Read the papers** - Many repos link to original research papers
4. **Join communities** - Most repos have active Discord/discussion boards
5. **Contribute back** - Share your learnings with the community

Remember: These repositories are learning tools. Our goal is to understand the concepts and build our own implementation, not just use pre-built solutions.