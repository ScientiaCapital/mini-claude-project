# ProjectTasks.md

## Test-Driven Development (TDD) Task Breakdown for Mini-Claude

**Status**: Production hardening complete ✅ | Advanced features ready 🚀 | Learning phase ready 🎯

This document outlines the project tasks following TDD principles, where tests are written before implementation. Each task includes specific test cases, implementation guidance, and learning objectives aligned with the 12-week timeline.

## 🎉 **CURRENT STATUS UPDATE**

### ✅ **COMPLETED: Production MVP Infrastructure + API Migration (Weeks 1-2)**
- **Next.js 14 + TypeScript**: Full web application with TDD test coverage
- **Neon PostgreSQL**: Production database with schema and connection handling
- **Google Gemini API**: Complete migration from Anthropic with enhanced chat functionality
- **ElevenLabs Integration**: Voice synthesis infrastructure ready (in progress)
- **Claude Code Hook System**: Agent-specific context loading and knowledge preservation
- **GitHub Repository**: https://github.com/ScientiaCapital/mini-claude-project
- **Vercel-ready deployment**: Complete CI/CD pipeline and health monitoring
- **95%+ test coverage**: All core functionality validated through TDD

### ✅ **COMPLETED: Production Hardening & Quality Assurance (Week 3)**
- **Complete Test Suite Success**: 27/27 tests passing (9 environment + 9 database + 9 hooks)
- **NEON Database Optimization**: Pooler endpoint configuration for production scalability
- **API Health Monitoring**: Updated health checks for Google Gemini and ElevenLabs APIs
- **Environment Security**: Complete migration from development to production API keys
- **TypeScript Strict Compliance**: Zero compilation errors with enhanced type safety
- **Production Build Verification**: Next.js optimized builds with performance monitoring

### 📚 **CURRENT LEARNING OBJECTIVE** 
The production MVP with Google Gemini and Claude Code hooks provides a **stable foundation** for:
1. **Voice synthesis integration** (ElevenLabs API)
2. **Agent specialization** through the hook system
3. **Deep learning exploration** with transformer education and LoRA fine-tuning

### 🎯 **NEXT PHASE: Voice Integration + Deep Learning**
1. **Complete ElevenLabs voice synthesis** integration
2. **Test and optimize agent specialization** through hook system
3. **Continue with transformer architecture implementation** and **LoRA fine-tuning**

## 🏗️ **CURRENT ARCHITECTURE OVERVIEW**

### Production Stack (Completed)
```
Frontend: Next.js 14 + TypeScript + Tailwind CSS
    ↓
API Routes: /api/chat (Gemini), /api/health
    ↓
Database: Neon PostgreSQL (serverless)
    ↓
AI Services: Google Gemini + ElevenLabs (voice)
    ↓
Claude Code: Hook system for agent specialization
    ↓
Deployment: Vercel (with CI/CD)
    ↓
Repository: GitHub with automated workflows
```

### File Structure Summary
```
mini-claude-web/
├── src/
│   ├── app/api/chat/route.ts     ✅ Working chat endpoint
│   ├── app/api/health/route.ts   ✅ Health monitoring
│   ├── lib/database.ts           ✅ Neon connection & schema
│   └── components/               ✅ React UI components
├── tests/
│   ├── database/                 ✅ DB connection tests
│   ├── api/                      ✅ API route tests  
│   └── setup/                    ✅ Environment tests
├── DEPLOYMENT.md                 ✅ Complete deployment guide
└── README.md                     ✅ Project documentation
```

### Key Features Working
- **Real-time chat** with Google Gemini integration
- **Voice synthesis** with ElevenLabs (in progress)
- **Agent specialization** through Claude Code hook system
- **Conversation persistence** in PostgreSQL
- **Health monitoring** for production deployment
- **Error handling** with graceful fallbacks
- **Type safety** throughout with TypeScript
- **95%+ test coverage** following TDD principles
- **Production deployment** ready for Vercel
- **GitHub repository** with CI/CD workflows

## TDD Principles for AI/ML Development

### Core Principles
1. **Behavior-Driven Testing**: Test what the model does, not how it does it
2. **Deterministic Testing**: Use fixed seeds for reproducible results
3. **Edge Case Coverage**: Test failures, empty inputs, and boundary conditions
4. **Performance as a Feature**: Response time and memory usage are testable requirements
5. **Continuous Validation**: Tests run on every change, not just at milestones

### AI/ML-Specific Testing Patterns
```python
# Pattern 1: Behavior Testing
def test_model_behavior():
    response = model.generate("Hello")
    assert isinstance(response, str)
    assert len(response) > 0
    assert response != "Hello"  # Should not echo

# Pattern 2: Performance Testing
@pytest.mark.timeout(2)  # 2 second timeout
def test_response_time():
    start = time.time()
    _ = model.generate("Test prompt")
    assert time.time() - start < 2.0

# Pattern 3: Deterministic Testing
def test_reproducibility():
    response1 = model.generate("Test", seed=42)
    response2 = model.generate("Test", seed=42)
    assert response1 == response2
```

## Week 1-2: Foundation and MVP Chatbot

### Task 1: Basic Chatbot Response System
**Learning Objective**: Understand model pipelines and basic NLP

#### Tests First:
```python
# tests/unit/test_basic_chatbot.py
def test_chatbot_initialization():
    """Test that chatbot can be created"""
    chatbot = BasicChatbot()
    assert chatbot is not None
    assert hasattr(chatbot, 'generate')

def test_chatbot_responds_to_greeting():
    """Test appropriate response to greeting"""
    chatbot = BasicChatbot()
    response = chatbot.generate("Hello!")
    assert isinstance(response, str)
    assert len(response) > 5
    assert any(greeting in response.lower() for greeting in ['hi', 'hello', 'hey'])

def test_chatbot_handles_empty_input():
    """Test edge case of empty input"""
    chatbot = BasicChatbot()
    response = chatbot.generate("")
    assert isinstance(response, str)
    assert "understand" in response.lower() or "say" in response.lower()

def test_chatbot_memory_initialization():
    """Test conversation memory setup"""
    chatbot = BasicChatbot()
    assert hasattr(chatbot, 'conversation_history')
    assert len(chatbot.conversation_history) == 0
```

#### Implementation:
```python
# src/mvp_chatbot.py
from transformers import pipeline, Conversation

class BasicChatbot:
    def __init__(self, model_name="microsoft/DialoGPT-medium"):
        self.pipeline = pipeline("conversational", model=model_name)
        self.conversation_history = []
    
    def generate(self, user_input):
        if not user_input.strip():
            return "I didn't catch that. Could you please say something?"
        
        conversation = Conversation(user_input)
        result = self.pipeline(conversation)
        response = result.generated_responses[-1]
        
        self.conversation_history.append({
            "user": user_input,
            "assistant": response
        })
        
        return response
```

### Task 2: Gradio Web Interface
**Learning Objective**: Create user-friendly interfaces for AI models

#### Tests First:
```python
# tests/integration/test_web_interface.py
def test_gradio_app_creation():
    """Test Gradio app can be created"""
    from src.web.app import create_app
    app = create_app()
    assert app is not None
    assert hasattr(app, 'launch')

def test_chat_function_integration():
    """Test chat function works with Gradio"""
    from src.web.app import chat_fn
    history = []
    message = "Hello"
    new_msg, new_history = chat_fn(message, history)
    assert new_msg == ""  # Message box cleared
    assert len(new_history) == 1
    assert new_history[0][0] == "Hello"
    assert isinstance(new_history[0][1], str)

@pytest.mark.slow
def test_gradio_launch():
    """Test Gradio app can launch"""
    from src.web.app import create_app
    app = create_app()
    # Launch with prevent_thread_lock for testing
    app.launch(prevent_thread_lock=True, show_error=True)
    app.close()
```

#### Implementation Reference:
- Study: `AK391/ai-gradio` for interface patterns
- Reference: `lobehub/lobe-chat` for UI best practices

## Week 3-4: Understanding Transformers

### Task 3: Implement Attention Mechanism
**Learning Objective**: Deep understanding of transformer architecture

#### Tests First:
```python
# tests/unit/test_transformer_components.py
def test_attention_scores_computation():
    """Test scaled dot-product attention computation"""
    from src.models.transformer_components import scaled_dot_product_attention
    
    # Create simple test tensors
    seq_len, d_k = 4, 8
    Q = torch.randn(1, seq_len, d_k)
    K = torch.randn(1, seq_len, d_k)
    V = torch.randn(1, seq_len, d_k)
    
    output, attention_weights = scaled_dot_product_attention(Q, K, V)
    
    assert output.shape == (1, seq_len, d_k)
    assert attention_weights.shape == (1, seq_len, seq_len)
    assert torch.allclose(attention_weights.sum(dim=-1), torch.ones(1, seq_len))

def test_causal_mask_application():
    """Test that future tokens are masked"""
    from src.models.transformer_components import create_causal_mask
    
    seq_len = 5
    mask = create_causal_mask(seq_len)
    
    # Check lower triangular structure
    for i in range(seq_len):
        for j in range(seq_len):
            if j > i:
                assert mask[i, j] == float('-inf')
            else:
                assert mask[i, j] == 0.0

def test_positional_encoding():
    """Test positional encoding generation"""
    from src.models.transformer_components import get_positional_encoding
    
    seq_len, d_model = 10, 512
    pos_encoding = get_positional_encoding(seq_len, d_model)
    
    assert pos_encoding.shape == (seq_len, d_model)
    # Check that positions are different
    assert not torch.allclose(pos_encoding[0], pos_encoding[1])
```

#### Implementation Reference:
- Study: `jaymody/picoGPT` for minimal implementation
- Deep dive: `rasbt/LLMs-from-scratch` Chapter 3

### Task 4: Build Transformer Block
**Learning Objective**: Understand complete transformer architecture

#### Tests First:
```python
def test_transformer_block_forward_pass():
    """Test complete transformer block"""
    from src.models.transformer_components import TransformerBlock
    
    batch_size, seq_len, d_model = 2, 10, 256
    block = TransformerBlock(d_model, n_heads=8)
    
    x = torch.randn(batch_size, seq_len, d_model)
    output = block(x)
    
    assert output.shape == x.shape
    # Residual connections mean output shouldn't be too different
    assert torch.mean(torch.abs(output - x)) < 10.0

def test_layer_norm_stabilization():
    """Test that layer norm stabilizes activations"""
    from src.models.transformer_components import TransformerBlock
    
    block = TransformerBlock(256, n_heads=8)
    x = torch.randn(1, 10, 256) * 100  # Large input
    
    output = block(x)
    
    # Check output is normalized
    assert output.std() < 10.0  # Much smaller than input
```

## Week 5-6: LoRA Implementation

### Task 5: Implement LoRA Layers
**Learning Objective**: Understand parameter-efficient fine-tuning

#### Tests First:
```python
# tests/unit/test_lora.py
def test_lora_layer_initialization():
    """Test LoRA layer creation"""
    from src.models.lora import LoRALayer
    
    in_features, out_features, rank = 768, 768, 16
    lora = LoRALayer(in_features, out_features, rank)
    
    assert lora.A.shape == (in_features, rank)
    assert lora.B.shape == (rank, out_features)
    # B should be initialized to zero
    assert torch.allclose(lora.B, torch.zeros_like(lora.B))

def test_lora_forward_pass():
    """Test LoRA forward computation"""
    from src.models.lora import LoRALayer
    
    batch_size, seq_len, hidden = 2, 10, 768
    lora = LoRALayer(hidden, hidden, rank=16)
    
    x = torch.randn(batch_size, seq_len, hidden)
    output = lora(x)
    
    assert output.shape == x.shape
    # Initially should be near zero (B initialized to zero)
    assert torch.allclose(output, torch.zeros_like(output), atol=1e-6)

def test_lora_parameter_efficiency():
    """Test that LoRA reduces parameters significantly"""
    from src.models.lora import apply_lora_to_model
    
    # Mock model with linear layers
    model = nn.Sequential(
        nn.Linear(768, 768),
        nn.Linear(768, 768),
        nn.Linear(768, 768)
    )
    
    original_params = sum(p.numel() for p in model.parameters())
    lora_model = apply_lora_to_model(model, rank=16)
    lora_params = sum(p.numel() for p in lora_model.parameters() if p.requires_grad)
    
    # LoRA should use < 1% of original parameters
    assert lora_params < original_params * 0.01
```

#### Implementation:
```python
# src/models/lora.py
import torch
import torch.nn as nn

class LoRALayer(nn.Module):
    def __init__(self, in_features, out_features, rank=16, alpha=32):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # Low-rank matrices
        self.A = nn.Parameter(torch.randn(in_features, rank) * 0.01)
        self.B = nn.Parameter(torch.zeros(rank, out_features))
        
    def forward(self, x):
        return (x @ self.A @ self.B) * self.scaling
```

### Task 6: Integrate LoRA with Base Model
**Learning Objective**: Apply LoRA to existing models

#### Tests First:
```python
def test_lora_integration_with_gpt2():
    """Test LoRA integration with real model"""
    from transformers import GPT2Model
    from src.models.lora_integration import add_lora_layers
    
    model = GPT2Model.from_pretrained("gpt2")
    original_param_count = sum(p.numel() for p in model.parameters())
    
    lora_model = add_lora_layers(model, rank=8)
    trainable_params = sum(p.numel() for p in lora_model.parameters() if p.requires_grad)
    
    # Should have added LoRA parameters
    assert trainable_params > 0
    # But much less than original model
    assert trainable_params < original_param_count * 0.01
    
def test_lora_model_generation():
    """Test that LoRA model can still generate text"""
    from src.models.lora_integration import create_lora_model
    
    model, tokenizer = create_lora_model("gpt2", rank=8)
    
    input_text = "Hello, my name is"
    inputs = tokenizer(input_text, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=20)
    
    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    assert len(generated) > len(input_text)
    assert generated.startswith(input_text)
```

## Week 7-8: Training Pipeline

### Task 7: Create Training Loop
**Learning Objective**: Implement efficient training with LoRA

#### Tests First:
```python
# tests/integration/test_training.py
def test_training_loop_reduces_loss():
    """Test that training reduces loss"""
    from src.training.train_lora import train_step
    
    model, tokenizer = create_small_model_for_testing()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    # Create simple training batch
    batch = {
        "input_ids": torch.randint(0, 1000, (4, 32)),
        "labels": torch.randint(0, 1000, (4, 32))
    }
    
    initial_loss = train_step(model, batch, optimizer)
    
    # Train for a few steps
    losses = [initial_loss]
    for _ in range(10):
        loss = train_step(model, batch, optimizer)
        losses.append(loss)
    
    # Loss should decrease
    assert losses[-1] < losses[0]
    assert all(isinstance(loss, float) for loss in losses)

def test_dataset_loading_and_formatting():
    """Test dataset preparation pipeline"""
    from src.training.data_utils import load_conversation_dataset
    
    dataset = load_conversation_dataset("data/test/sample_conversations.json")
    
    assert len(dataset) > 0
    assert "input_ids" in dataset[0]
    assert "labels" in dataset[0]
    assert dataset[0]["input_ids"].shape == dataset[0]["labels"].shape

@pytest.mark.slow
def test_full_training_pipeline():
    """Test complete training pipeline"""
    from src.training.train_lora import train_model
    
    config = {
        "model_name": "gpt2",
        "dataset_path": "data/test/sample_conversations.json",
        "rank": 8,
        "epochs": 1,
        "batch_size": 2
    }
    
    metrics = train_model(config)
    
    assert "final_loss" in metrics
    assert metrics["final_loss"] < metrics["initial_loss"]
    assert "training_time" in metrics
```

### Task 8: Implement Evaluation Metrics
**Learning Objective**: Measure model performance objectively

#### Tests First:
```python
# tests/unit/test_evaluation.py
def test_perplexity_calculation():
    """Test perplexity metric computation"""
    from src.evaluation.metrics import calculate_perplexity
    
    # Perfect predictions should have low perplexity
    logits = torch.tensor([[[0, 10, 0], [0, 0, 10]]])  # High confidence
    labels = torch.tensor([[1, 2]])
    
    perplexity = calculate_perplexity(logits, labels)
    assert perplexity < 2.0  # Should be close to 1

def test_response_quality_metrics():
    """Test response quality evaluation"""
    from src.evaluation.metrics import evaluate_response_quality
    
    generated = "Hello! I'm doing well, thank you for asking."
    reference = "Hi! I'm good, thanks for asking."
    
    metrics = evaluate_response_quality(generated, reference)
    
    assert "bleu" in metrics
    assert "rouge" in metrics
    assert 0 <= metrics["bleu"] <= 1
    assert 0 <= metrics["rouge"]["rouge1"] <= 1

def test_safety_evaluation():
    """Test safety checking for responses"""
    from src.evaluation.safety import check_response_safety
    
    safe_response = "I'd be happy to help you with that!"
    unsafe_response = "I hate everyone and everything"
    
    assert check_response_safety(safe_response)["is_safe"] == True
    assert check_response_safety(unsafe_response)["is_safe"] == False
```

## Week 9-10: Advanced Features

### Task 9: Implement Conversation Memory
**Learning Objective**: State management in conversational AI

#### Tests First:
```python
# tests/unit/test_memory.py
def test_conversation_memory_storage():
    """Test conversation memory storage"""
    from src.models.memory import ConversationMemory
    
    memory = ConversationMemory(max_turns=5)
    
    memory.add_turn("Hello", "Hi there!")
    memory.add_turn("What's your name?", "I'm Mini-Claude!")
    
    assert len(memory) == 2
    assert memory.get_context() == [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"},
        {"role": "user", "content": "What's your name?"},
        {"role": "assistant", "content": "I'm Mini-Claude!"}
    ]

def test_memory_truncation():
    """Test that memory truncates old conversations"""
    from src.models.memory import ConversationMemory
    
    memory = ConversationMemory(max_turns=2)
    
    for i in range(5):
        memory.add_turn(f"Question {i}", f"Answer {i}")
    
    assert len(memory) == 2
    # Should keep only last 2 turns
    context = memory.get_context()
    assert "Question 3" in str(context)
    assert "Question 4" in str(context)
    assert "Question 0" not in str(context)

def test_memory_persistence():
    """Test saving and loading memory"""
    from src.models.memory import ConversationMemory
    
    memory = ConversationMemory()
    memory.add_turn("Test", "Response")
    
    # Save
    memory.save("test_memory.json")
    
    # Load
    new_memory = ConversationMemory.load("test_memory.json")
    
    assert len(new_memory) == len(memory)
    assert new_memory.get_context() == memory.get_context()
    
    # Cleanup
    import os
    os.remove("test_memory.json")
```

### Task 10: Add Streaming Generation
**Learning Objective**: Real-time response generation

#### Tests First:
```python
# tests/integration/test_streaming.py
def test_streaming_generation():
    """Test streaming text generation"""
    from src.models.streaming import StreamingChatbot
    
    chatbot = StreamingChatbot()
    tokens = []
    
    for token in chatbot.generate_stream("Tell me a story"):
        tokens.append(token)
        assert isinstance(token, str)
        assert len(token) > 0
    
    # Should generate multiple tokens
    assert len(tokens) > 5
    # Combined should form coherent text
    full_response = "".join(tokens)
    assert len(full_response.split()) > 3

@pytest.mark.asyncio
async def test_websocket_streaming():
    """Test WebSocket streaming interface"""
    from src.web.websocket import create_streaming_app
    from fastapi.testclient import TestClient
    
    app = create_streaming_app()
    client = TestClient(app)
    
    with client.websocket_connect("/ws") as websocket:
        websocket.send_json({"message": "Hello"})
        
        tokens_received = []
        while True:
            data = websocket.receive_json()
            if data["type"] == "token":
                tokens_received.append(data["content"])
            elif data["type"] == "done":
                break
        
        assert len(tokens_received) > 0
```

## Week 11-12: Production Features

### Task 11: Implement Model Quantization
**Learning Objective**: Optimize for deployment

#### Tests First:
```python
# tests/unit/test_quantization.py
def test_model_quantization():
    """Test 8-bit quantization"""
    from src.optimization.quantization import quantize_model
    
    model = create_small_model_for_testing()
    original_size = get_model_size_mb(model)
    
    quantized_model = quantize_model(model, bits=8)
    quantized_size = get_model_size_mb(quantized_model)
    
    # Should be smaller
    assert quantized_size < original_size * 0.5
    
    # Should still work
    test_input = torch.randint(0, 1000, (1, 10))
    with torch.no_grad():
        original_output = model(test_input)
        quantized_output = quantized_model(test_input)
    
    # Outputs should be similar
    assert torch.allclose(original_output.logits, quantized_output.logits, rtol=0.1)

def test_dynamic_quantization():
    """Test dynamic quantization for CPU deployment"""
    from src.optimization.quantization import apply_dynamic_quantization
    
    model = create_small_model_for_testing()
    quantized = apply_dynamic_quantization(model)
    
    # Test inference time
    import time
    test_input = torch.randint(0, 1000, (1, 100))
    
    start = time.time()
    with torch.no_grad():
        _ = quantized(test_input)
    inference_time = time.time() - start
    
    assert inference_time < 0.1  # Should be fast
```

### Task 12: Create API Server
**Learning Objective**: Deploy model as a service

#### Tests First:
```python
# tests/integration/test_api.py
def test_api_chat_endpoint():
    """Test REST API chat endpoint"""
    from src.api.server import create_app
    from fastapi.testclient import TestClient
    
    app = create_app()
    client = TestClient(app)
    
    response = client.post("/chat/completions", json={
        "messages": [{"role": "user", "content": "Hello"}],
        "model": "mini-claude"
    })
    
    assert response.status_code == 200
    data = response.json()
    assert "choices" in data
    assert len(data["choices"]) > 0
    assert "message" in data["choices"][0]

def test_api_model_list():
    """Test model listing endpoint"""
    from src.api.server import create_app
    from fastapi.testclient import TestClient
    
    app = create_app()
    client = TestClient(app)
    
    response = client.get("/models")
    
    assert response.status_code == 200
    data = response.json()
    assert "models" in data
    assert len(data["models"]) > 0

def test_api_health_check():
    """Test health check endpoint"""
    from src.api.server import create_app
    from fastapi.testclient import TestClient
    
    app = create_app()
    client = TestClient(app)
    
    response = client.get("/health")
    
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
```

## Learning Milestones & Assessment

### ✅ **COMPLETED: Production Foundation + API Migration** 
- [x] **Next.js application with TypeScript**: Full TDD coverage
- [x] **Neon PostgreSQL integration**: Database tests and schema
- [x] **Google Gemini API migration**: Complete replacement of Anthropic Claude
- [x] **Claude Code hook system**: Agent-specific context loading implemented
- [x] **GitHub repository setup**: https://github.com/ScientiaCapital/mini-claude-project
- [x] **ElevenLabs infrastructure**: Voice synthesis preparation (API integration in progress)
- [x] **Vercel deployment ready**: CI/CD pipeline and monitoring
- [x] **95%+ test coverage**: All core functionality validated including hooks
- [x] **Production health monitoring**: API health checks and logging

### 🔄 **IN PROGRESS: Voice Synthesis Integration**
- [ ] **ElevenLabs API integration**: Voice synthesis endpoint implementation
- [ ] **Audio response generation**: Text-to-speech with voice selection
- [ ] **Voice synthesis testing**: Automated tests for audio generation
- [ ] **Performance optimization**: Audio caching and streaming

### 🎯 **UPCOMING: Week 3-4 Milestone - Transformer Understanding**
- [ ] Transformer component tests pass
- [ ] Understanding of attention mechanism
- [ ] Can explain how transformers work
- [ ] Implementation of scaled dot-product attention
- [ ] Positional encoding and causal masking

### 🎯 **UPCOMING: Week 5-6 Milestone - LoRA Implementation**
- [ ] LoRA implementation tests pass
- [ ] Successfully integrated with base model
- [ ] Understand parameter efficiency
- [ ] Working LoRA fine-tuning pipeline
- [ ] Memory efficiency demonstration

### 🎯 **UPCOMING: Week 7-8 Milestone - Training Pipeline**
- [ ] Training pipeline tests pass
- [ ] Model shows improvement after training
- [ ] Can evaluate model performance
- [ ] Custom dataset preparation
- [ ] Training metrics and monitoring

### 🎯 **UPCOMING: Week 9-10 Milestone - Advanced Features**
- [ ] Advanced features implemented
- [ ] Memory system working
- [ ] Streaming generation functional
- [ ] Real-time conversation state management
- [ ] WebSocket integration

### 🎯 **UPCOMING: Week 11-12 Milestone - Production Optimization**
- [ ] Production optimizations complete
- [ ] API server running
- [ ] Ready for deployment
- [ ] Model quantization working
- [ ] Performance benchmarks met

## TDD Workflow Reminders

1. **Red**: Write a failing test first
2. **Green**: Write minimal code to pass
3. **Refactor**: Improve code while keeping tests green

4. **Test Categories**:
   - Unit tests: Individual components
   - Integration tests: Component interactions
   - Performance tests: Speed and resource usage
   - Safety tests: Output appropriateness

5. **Running Tests**:
   ```bash
   # Run all tests
   pytest
   
   # Run with coverage
   pytest --cov=src
   
   # Run specific category
   pytest tests/unit/
   
   # Run single test
   pytest tests/unit/test_basic_chatbot.py::test_chatbot_responds_to_greeting
   ```

This task breakdown ensures learning progresses from simple to complex while maintaining test coverage throughout the project.