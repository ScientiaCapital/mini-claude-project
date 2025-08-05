# ProjectTasks.md

## Test-Driven Development (TDD) Task Breakdown for Mini-Claude

**Status**: Production MVP complete ✅ | Voice integration in progress 🎯

This document outlines the project tasks following TDD principles, where tests are written before implementation. Each task includes specific test cases, implementation guidance, and learning objectives aligned with the 12-week timeline.

## 🎉 **CURRENT STATUS UPDATE**

### ✅ **COMPLETED: Production MVP with Real Database**
- **Next.js 14 + TypeScript**: Full web application with TDD test coverage
- **Neon PostgreSQL**: All database tests passing with real connections
- **Google Gemini API**: Chat functionality fully integrated
- **Voice Synthesis Module**: ElevenLabs implementation complete with tests
- **pgvector Extension**: Installed with agent_memory table ready
- **GitHub Repository**: https://github.com/ScientiaCapital/mini-claude-project
- **Test Coverage**: 95%+ maintained with no mocks

### 🎯 **IMMEDIATE NEXT STEPS**
1. **Vector Database Tests**: Implement and test pgvector functionality
2. **Voice API Integration**: Add voice synthesis to chat endpoint
3. **Vercel Deployment**: Deploy with production environment variables
4. **Streaming Responses**: Add real-time chat capabilities

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
│   ├── app/api/chat/route.ts     ✅ Google Gemini chat
│   ├── app/api/health/route.ts   ✅ Health monitoring
│   ├── lib/database.ts           ✅ Real Neon connection
│   ├── lib/voice-synthesis.ts    ✅ ElevenLabs module
│   └── components/               ✅ React UI
├── tests/
│   ├── database/                 ✅ All tests passing
│   ├── voice-synthesis/          ✅ Voice tests passing
│   └── api/                      🎯 Ready for voice
├── scripts/init-db.mjs           ✅ Database setup
└── .env.local                    🔑 Real API keys
```

### Key Features Working
- **Real-time chat** with Google Gemini ✅
- **Database tests** with real Neon connection ✅
- **Voice synthesis module** fully tested ✅
- **pgvector** installed and ready ✅
- **Health monitoring** endpoint working ✅
- **Type safety** with TypeScript ✅
- **95%+ test coverage** maintained ✅

### Ready to Implement
- **Voice in chat API** (module ready)
- **Vector similarity search** (database ready)
- **Vercel deployment** (config ready)
- **Streaming responses** (architecture ready)

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

## Current Tasks: Vector Tests and Voice Integration

### Task 1: Implement Vector Database Tests
**Status**: Ready to implement 🎯

#### Tests to Write:
```typescript
// tests/database/vector.test.ts
describe('pgvector functionality', () => {
  test('should store embeddings in agent_memory', async () => {
    const embedding = new Array(1536).fill(0.1)
    const result = await storeAgentMemory({
      agent_type: 'general-purpose',
      content: 'Test memory',
      embedding
    })
    expect(result.id).toBeDefined()
  })

  test('should perform similarity search', async () => {
    // Store test embeddings
    const embedding1 = new Array(1536).fill(0.1)
    const embedding2 = new Array(1536).fill(0.9)
    
    await storeAgentMemory({ content: 'Similar', embedding: embedding1 })
    await storeAgentMemory({ content: 'Different', embedding: embedding2 })
    
    // Search for similar
    const results = await searchSimilar(embedding1, 5)
    expect(results[0].content).toBe('Similar')
  })
})
```

### Task 2: Integrate Voice Synthesis in Chat API
**Status**: Module ready, integration needed 🎯

#### Implementation Plan:
```typescript
// src/app/api/chat/route.ts updates
export async function POST(request: Request) {
  const { messages, voice_enabled, voice_id } = await request.json()
  
  // Get Gemini response
  const reply = await generateChatResponse(messages)
  
  // Generate voice if enabled
  let audio_url = undefined
  if (voice_enabled && reply) {
    const voiceResponse = await synthesizeVoice({
      text: reply,
      voice_id: voice_id || 'default_voice_id'
    })
    audio_url = voiceResponse.audio_url
  }
  
  return NextResponse.json({
    reply,
    audio_url,
    message_id: generateId(),
    conversation_id: conversationId
  })
}
```

### Task 3: Deploy to Vercel
**Status**: Ready to deploy 🎯

#### Deployment Checklist:
```bash
# 1. Set environment variables in Vercel
NEON_DATABASE_URL=postgresql://...
GOOGLE_API_KEY=your-key
ELEVENLABS_API_KEY=your-key
NEXTAUTH_SECRET=your-secret

# 2. Deploy commands
cd mini-claude-web
vercel          # Preview deployment
vercel --prod   # Production deployment

# 3. Verify deployment
curl https://your-app.vercel.app/api/health
```

#### Post-Deployment Tests:
- Health endpoint returns 200
- Chat API works with real Gemini
- Database connection successful
- Voice synthesis ready to test

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

## Future Learning Tasks (Educational)

### Understanding Transformers
**Resources**: `rasbt/LLMs-from-scratch`, `jaymody/picoGPT`
- Study self-attention mechanism
- Implement scaled dot-product attention
- Build transformer blocks
- Create positional encodings

### LoRA Fine-tuning Experiments
**Resources**: `huggingface/peft`, Microsoft LoRA paper
- Implement LoRA layers
- Test parameter efficiency
- Fine-tune small models
- Measure performance gains

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

## Progress Summary

### ✅ **COMPLETED: Production MVP** 
- [x] **Next.js + TypeScript**: Full application with TDD
- [x] **Neon PostgreSQL**: All tests passing with real connection
- [x] **Google Gemini API**: Chat fully integrated
- [x] **Voice Synthesis Module**: ElevenLabs implementation complete
- [x] **pgvector Extension**: Installed with agent_memory table
- [x] **Database Schema**: users, conversations, messages tables
- [x] **Test Coverage**: 95%+ maintained throughout
- [x] **No Mocks**: All tests use real connections

### 🎯 **THIS WEEK: Integration & Deployment**
- [ ] **Vector Tests**: Implement pgvector functionality tests
- [ ] **Voice in Chat API**: Integrate voice synthesis module
- [ ] **Vercel Deployment**: Deploy with environment variables
- [ ] **Streaming Responses**: Add real-time chat support

### 📚 **FUTURE: Educational Components**
- [ ] **Transformer Understanding**: Study architecture
- [ ] **LoRA Experiments**: Parameter-efficient fine-tuning
- [ ] **RAG Integration**: Knowledge base augmentation

## Current Development Workflow

### Running Tests
```bash
# Run all tests
npm test

# Run specific tests
npm test tests/database/
npm test tests/voice-synthesis/

# Run with coverage
npm run test:coverage
```

### TDD Principles Applied
1. **Write test first** (Red phase)
2. **Implement minimum code** (Green phase)  
3. **Refactor with confidence** (Tests stay green)
4. **Real connections only** (No mocks in production)

### Next Development Cycle
1. Write vector database tests
2. Implement pgvector functions
3. Add voice to chat API
4. Deploy and verify