import os
from dotenv import load_dotenv
from anthropic import Anthropic

load_dotenv()
client = Anthropic()

SYSTEM_PROMPT = """You are a helpful AI assistant. Be concise but friendly.
Remember everything the user tells you in this conversation."""

def chat():
    """
    A simple CLI chatbot with conversation memory.
    
    Key insight: Claude has NO memory between API calls.
    You create "memory" by sending the full conversation history
    in every request. This is how all LLM chat apps work.
    """
    
    # This list grows with every exchange — it IS the memory
    conversation_history = []
    
    print("Claude CLI Chat — type 'quit' to exit, 'history' to see the conversation\n")
    print("-" * 50)
    
    while True:
        user_input = input("\nYou: ").strip()
        
        if user_input.lower() == "quit":
            print("Goodbye!")
            break
            
        if user_input.lower() == "history":
            print("\n--- Conversation so far ---")
            for msg in conversation_history:
                role = "You" if msg["role"] == "user" else "Claude"
                print(f"{role}: {msg['content'][:100]}...")
            continue
            
        if not user_input:
            continue
        
        # Add user message to history
        conversation_history.append({
            "role": "user",
            "content": user_input
        })
        
        # Send the ENTIRE history every time — this is how Claude remembers
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            messages=conversation_history  # <-- full history, not just latest message
        )
        
        assistant_reply = response.content[0].text
        
        # Add Claude's reply to history too
        conversation_history.append({
            "role": "assistant",
            "content": assistant_reply
        })
        
        print(f"\nClaude: {assistant_reply}")
        print(f"(tokens used this turn: {response.usage.input_tokens} in, {response.usage.output_tokens} out)")

if __name__ == "__main__":
    chat()
