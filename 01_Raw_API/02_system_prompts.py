import os
from dotenv import load_dotenv
from anthropic import Anthropic

load_dotenv()
client = Anthropic()

# A system prompt defines Claude's persona, role, and rules
# This is the most powerful prompting tool you have
SYSTEM_PROMPT = """You are a business development executive for Distributed Antenna Systems (DAS) 
for small and medium businesses. 

Your style:
- Be concise and direct — no fluff
- Always give concrete, actionable advice
- Use bullet points for lists
- When relevant, mention cost or time estimates
- Never recommend tools you're not confident about

Your audience: highrise building owners with no technical background."""

def ask(question):
    """Helper function to ask Claude a question with our system prompt."""
    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        system=SYSTEM_PROMPT,   # <-- The system prompt goes here, separate from messages
        messages=[
            {"role": "user", "content": question}
        ]
    )
    return message.content[0].text

# Try the same question with different system prompts and compare results
print("=== With business development persona ===\n")
print(ask("How should I approach the customer with a cellular service deployment at their buildings by installing a multi-carrier DAS?"))

print("\n\n=== Change the system prompt above to a different persona and re-run! ===")
print("Try: a skeptical CTO, a startup founder, a cost-conscious accountant")
