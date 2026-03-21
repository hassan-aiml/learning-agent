import os
from dotenv import load_dotenv
from anthropic import Anthropic

# Load your API key from .env
load_dotenv()

# Create the client — this is your connection to Claude
client = Anthropic()

# Send a message and get a response
message = client.messages.create(
    model="claude-sonnet-4-6",   # The model to use
    max_tokens=1024,              # Max length of the response
    messages=[
        {
            "role": "user",
            "content": "What are 3 practical uses of AI agents for small businesses?"
        }
    ]
)

# The response lives in message.content — it's a list of content blocks
# For text responses, we want the first block's text
print(message.content[0].text)

# Bonus: print token usage so you understand costs
print(f"\n--- Usage ---")
print(f"Input tokens:  {message.usage.input_tokens}")
print(f"Output tokens: {message.usage.output_tokens}")
