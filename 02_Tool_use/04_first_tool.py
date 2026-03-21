import os
from dotenv import load_dotenv
from anthropic import Anthropic

load_dotenv()
client = Anthropic()

# ── 1. Define your Python function (the actual logic) ──────────────────────────
def get_weather(city: str) -> str:
    """Fake weather data — in a real app you'd call a weather API here."""
    weather_data = {
        "london":   "12°C, overcast",
        "new york": "22°C, sunny",
        "tokyo":    "18°C, light rain",
        "sydney":   "25°C, clear skies",
    }
    return weather_data.get(city.lower(), f"No data available for {city}")


# ── 2. Describe the tool to Claude (Claude reads this to know what exists) ─────
tools = [
    {
        "name": "get_weather",
        "description": "Get the current weather for a given city. Use this whenever the user asks about weather.",
        "input_schema": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "The name of the city, e.g. London"
                }
            },
            "required": ["city"]
        }
    }
]


# ── 3. Send message + tools to Claude ─────────────────────────────────────────
user_message = "What's the weather like in Tokyo right now?"
print(f"User: {user_message}\n")

response = client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=1024,
    tools=tools,                # <-- hand Claude the tool descriptions
    messages=[{"role": "user", "content": user_message}]
)

print(f"Stop reason: {response.stop_reason}")  # will print 'tool_use'
print(f"Response content: {response.content}\n")


# ── 4. Check if Claude wants to use a tool ────────────────────────────────────
if response.stop_reason == "tool_use":

    # Find the tool_use block in the response
    tool_use_block = next(b for b in response.content if b.type == "tool_use")

    tool_name = tool_use_block.name          # "get_weather"
    tool_input = tool_use_block.input        # {"city": "Tokyo"}
    tool_use_id = tool_use_block.id          # needed to send the result back

    print(f"Claude wants to call: {tool_name}")
    print(f"With arguments: {tool_input}\n")

    # ── 5. YOU run the actual function ────────────────────────────────────────
    if tool_name == "get_weather":
        result = get_weather(tool_input["city"])
        print(f"Function returned: {result}\n")

    # ── 6. Send the result back so Claude can form its final response ──────────
    final_response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        tools=tools,
        messages=[
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": response.content},  # Claude's tool_use block
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": tool_use_id,   # must match the block id
                        "content": result
                    }
                ]
            }
        ]
    )

    print(f"Claude: {final_response.content[0].text}")
