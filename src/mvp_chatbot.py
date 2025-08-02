"""
MVP Chatbot Implementation
Following TDD principles - this is the minimal implementation
to make our tests pass.
"""
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
import os
from huggingface_hub import snapshot_download


class MVPChatbot:
    """
    Minimal Viable Product chatbot using DialoGPT.
    This is our starting point - a working chatbot that we can enhance.
    """
    
    def __init__(self, model_name="microsoft/DialoGPT-medium"):
        """
        Initialize the chatbot with a pre-trained model.
        
        Args:
            model_name: Name of the model to use (default: DialoGPT-medium)
        """
        self.model_name = model_name
        self.conversation_history = []
        
        # Initialize the model and tokenizer
        print(f"Loading model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        
        # Add padding token for newer transformers versions
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def generate(self, user_input: str) -> str:
        """
        Generate a response to user input.
        
        Args:
            user_input: The user's message
            
        Returns:
            The chatbot's response
        """
        # Handle empty or whitespace input
        if not user_input.strip():
            return "I didn't catch that. Could you please say something?"
        
        # Encode the input
        # For DialoGPT, we need to append the EOS token
        new_user_input_ids = self.tokenizer.encode(
            user_input + self.tokenizer.eos_token, 
            return_tensors='pt'
        )
        
        # Generate response
        # We use attention mask to handle padding properly
        with torch.no_grad():
            # Create attention mask
            attention_mask = new_user_input_ids.ne(self.tokenizer.pad_token_id).long()
            
            # Generate
            chat_history_ids = self.model.generate(
                new_user_input_ids,
                max_length=1000,
                attention_mask=attention_mask,
                pad_token_id=self.tokenizer.eos_token_id,
                temperature=0.9,  # Add some randomness
                do_sample=True,   # Enable sampling for variety
                top_p=0.95,       # Nucleus sampling
                repetition_penalty=1.2  # Reduce repetition
            )
        
        # Decode the response
        # Skip the input tokens to get only the response
        response_ids = chat_history_ids[:, new_user_input_ids.shape[-1]:]
        response = self.tokenizer.decode(response_ids[0], skip_special_tokens=True)
        
        # If response is empty, provide a default
        if not response.strip():
            response = "I'm here to chat! What would you like to talk about?"
        
        # Save to conversation history
        self.conversation_history.append({
            "user": user_input,
            "assistant": response
        })
        
        return response


# Quick test if run directly
if __name__ == "__main__":
    print("Testing MVP Chatbot...")
    bot = MVPChatbot()
    
    test_inputs = [
        "Hello!",
        "How are you?",
        "What's your name?",
        "",  # Test empty input
    ]
    
    for test_input in test_inputs:
        print(f"\nUser: {test_input}")
        response = bot.generate(test_input)
        print(f"Bot: {response}")