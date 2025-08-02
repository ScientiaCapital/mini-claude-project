"""
Test-Driven Development: MVP Chatbot Tests
These tests define the behavior of our minimal viable chatbot
before we implement it. Following TDD: Red -> Green -> Refactor
"""
import pytest
import time
from unittest.mock import Mock, patch


def test_chatbot_class_exists():
    """Test that MVPChatbot class can be imported"""
    try:
        from src.mvp_chatbot import MVPChatbot
        assert MVPChatbot is not None
    except ImportError:
        pytest.fail("MVPChatbot class should be importable from src.mvp_chatbot")


def test_chatbot_initialization():
    """Test chatbot initializes with correct attributes"""
    from src.mvp_chatbot import MVPChatbot
    
    bot = MVPChatbot()
    
    # Should have essential attributes
    assert hasattr(bot, 'model'), "Chatbot should have a model attribute"
    assert hasattr(bot, 'tokenizer'), "Chatbot should have a tokenizer attribute"
    assert hasattr(bot, 'model_name'), "Chatbot should have a model_name attribute"
    
    # Default model should be DialoGPT-medium
    assert bot.model_name == "microsoft/DialoGPT-medium", \
        "Default model should be DialoGPT-medium"


def test_chatbot_has_generate_method():
    """Test chatbot has a generate method"""
    from src.mvp_chatbot import MVPChatbot
    
    bot = MVPChatbot()
    assert hasattr(bot, 'generate'), "Chatbot should have a generate method"
    assert callable(bot.generate), "generate should be a callable method"


def test_chatbot_responds_to_hello():
    """Test chatbot generates appropriate response to greeting"""
    from src.mvp_chatbot import MVPChatbot
    
    bot = MVPChatbot()
    response = bot.generate("Hello!")
    
    # Basic response validation
    assert isinstance(response, str), "Response should be a string"
    assert len(response) > 0, "Response should not be empty"
    assert response != "Hello!", "Response should not just echo the input"
    
    # Response should be conversational
    assert len(response.split()) >= 2, "Response should be at least 2 words"


def test_chatbot_handles_empty_input():
    """Test edge case of empty input"""
    from src.mvp_chatbot import MVPChatbot
    
    bot = MVPChatbot()
    response = bot.generate("")
    
    assert isinstance(response, str), "Response should be a string even for empty input"
    assert len(response) > 0, "Should provide a response for empty input"
    
    # Should indicate confusion or ask for input
    assert any(word in response.lower() for word in 
              ["understand", "say", "sorry", "pardon", "repeat", "didn't"]), \
        "Response should indicate the bot didn't understand"


def test_chatbot_handles_whitespace_input():
    """Test edge case of whitespace-only input"""
    from src.mvp_chatbot import MVPChatbot
    
    bot = MVPChatbot()
    response = bot.generate("   \n\t   ")
    
    assert isinstance(response, str), "Response should be a string"
    assert len(response) > 0, "Should provide a response for whitespace input"


def test_chatbot_memory_initialization():
    """Test conversation memory is properly initialized"""
    from src.mvp_chatbot import MVPChatbot
    
    bot = MVPChatbot()
    
    assert hasattr(bot, 'conversation_history'), \
        "Chatbot should have conversation_history attribute"
    assert isinstance(bot.conversation_history, list), \
        "conversation_history should be a list"
    assert len(bot.conversation_history) == 0, \
        "conversation_history should start empty"


def test_chatbot_saves_conversation():
    """Test that conversations are saved to history"""
    from src.mvp_chatbot import MVPChatbot
    
    bot = MVPChatbot()
    user_input = "What's your name?"
    response = bot.generate(user_input)
    
    # Check conversation was saved
    assert len(bot.conversation_history) == 1, \
        "Conversation history should have one entry"
    
    entry = bot.conversation_history[0]
    assert "user" in entry, "History entry should have user input"
    assert "assistant" in entry, "History entry should have assistant response"
    assert entry["user"] == user_input, "User input should be saved correctly"
    assert entry["assistant"] == response, "Assistant response should be saved correctly"


@pytest.mark.timeout(2)  # 2 second timeout
def test_chatbot_response_time():
    """Test that response time is under 2 seconds"""
    from src.mvp_chatbot import MVPChatbot
    
    bot = MVPChatbot()
    
    start_time = time.time()
    _ = bot.generate("Hello, how are you today?")
    end_time = time.time()
    
    response_time = end_time - start_time
    assert response_time < 2.0, f"Response took {response_time}s, should be < 2s"


def test_chatbot_generates_different_responses():
    """Test that chatbot doesn't always give the same response"""
    from src.mvp_chatbot import MVPChatbot
    
    bot = MVPChatbot()
    
    # Generate multiple responses to the same input
    responses = []
    for _ in range(3):
        response = bot.generate("Tell me something interesting")
        responses.append(response)
    
    # At least some responses should be different
    # (Note: this might occasionally fail due to randomness, but it's unlikely)
    unique_responses = set(responses)
    assert len(unique_responses) > 1, \
        "Chatbot should generate varied responses, not always the same"


def test_chatbot_custom_model_initialization():
    """Test that chatbot can be initialized with a custom model"""
    from src.mvp_chatbot import MVPChatbot
    
    # Test with a different model
    custom_model_name = "microsoft/DialoGPT-small"
    bot = MVPChatbot(model_name=custom_model_name)
    
    assert bot.model_name == custom_model_name, \
        f"Chatbot should use the specified model: {custom_model_name}"


def test_chatbot_handles_long_input():
    """Test chatbot handles long input gracefully"""
    from src.mvp_chatbot import MVPChatbot
    
    bot = MVPChatbot()
    
    # Create a long input (but not too long to break things)
    long_input = "Hello " * 50  # 50 "Hello"s
    response = bot.generate(long_input)
    
    assert isinstance(response, str), "Should handle long input"
    assert len(response) > 0, "Should generate a response for long input"


def test_chatbot_conversation_continuity():
    """Test that chatbot can handle multi-turn conversations"""
    from src.mvp_chatbot import MVPChatbot
    
    bot = MVPChatbot()
    
    # First turn
    response1 = bot.generate("My name is Alice")
    assert len(response1) > 0
    
    # Second turn - bot should potentially remember context
    response2 = bot.generate("What's my name?")
    assert len(response2) > 0
    
    # Check both turns are in history
    assert len(bot.conversation_history) == 2