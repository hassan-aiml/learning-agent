import os
import json
from dotenv import load_dotenv
from anthropic import Anthropic

load_dotenv()
client = Anthropic()

# ── This is the real agentic loop ─────────────────────────────────────────────
# Claude can call tools MULTIPLE times in sequence to complete a task.
# The loop runs until Claude stops asking for tools and gives a final answer.

def get_weather(city: str) -> str:
    data = {"london": "12°C, overcast", "paris": "15°C, sunny", "berlin": "8°C, cloudy"}
    return data.get(city.lower(), f"No data for {city}")

def get_exchange_rate(from_currency: str, to_currency: str) -> str:
    rates = {"USD_EUR": 0.92, "EUR_USD": 1.09, "GBP_USD": 1.27, "USD_GBP": 0.79}
    key = f"{from_currency.upper()}_{to_currency.upper()}"
    rate = rates.get(key, None)
    if rate:
        return f"1 {from_currency.upper()} = {rate} {to_currency.upper()}"
    return f"Rate not found for {from_currency}/{to_currency}"

def search_flights(origin: str, destination: str) -> str:
    return f"Found 3 flights from {origin} to {destination}: £299 (7am), £349 (12pm), £199 (11pm)"

def get_population(city: str) -> str:
    data = {"london": "~30 million", "paris": "~12 million", "berlin": "~15 million"}
    return data.get(city.lower(), f"No data for {city}")

tools = [
    {
        "name": "get_weather",
        "description": "Get current weather for a city.",
        "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}
    },
    {
        "name": "get_exchange_rate",
        "description": "Get exchange rate between two currencies.",
        "input_schema": {
            "type": "object",
            "properties": {
                "from_currency": {"type": "string", "description": "e.g. USD"},
                "to_currency":   {"type": "string", "description": "e.g. EUR"}
            },
            "required": ["from_currency", "to_currency"]
        }
    },
    {
        "name": "search_flights",
        "description": "Search for available flights between two cities.",
        "input_schema": {
            "type": "object",
            "properties": {
                "origin":      {"type": "string"},
                "destination": {"type": "string"}
            },
            "required": ["origin", "destination"]
        }
    },
    {
        "name": "get_population",
        "description": "Get population for a city.",
        "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}
    }
]

def run_tool(name: str, inputs: dict) -> str:
    if name == "get_weather":       return get_weather(inputs["city"])
    if name == "get_exchange_rate": return get_exchange_rate(inputs["from_currency"], inputs["to_currency"])
    if name == "search_flights":    return search_flights(inputs["origin"], inputs["destination"])
    if name == "get_population":       return get_population(inputs["city"])
    return f"Unknown tool: {name}"


# ── The agentic loop ───────────────────────────────────────────────────────────
def run_agent(user_message: str):
    print(f"User: {user_message}\n")

    messages = [{"role": "user", "content": user_message}]
    step = 1

    while True:
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=2048,
            tools=tools,
            messages=messages
        )

        # Claude is done — print final answer and exit loop
        if response.stop_reason == "end_turn":
            print(f"Claude: {response.content[0].text}")
            break

        # Claude wants to use tools — could be multiple in one response
        if response.stop_reason == "tool_use":
            # Add Claude's response to history
            messages.append({"role": "assistant", "content": response.content})

            # Process ALL tool calls Claude requested
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    print(f"Step {step}: Claude calls {block.name}({block.input})")
                    result = run_tool(block.name, block.input)
                    print(f"         Result: {result}")
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result
                    })
                    step += 1

            # Send all results back in one message
            messages.append({"role": "user", "content": tool_results})

        else:
            # Unexpected stop reason
            print(f"Unexpected stop reason: {response.stop_reason}")
            break

    print(f"\nTotal steps: {step - 1} tool calls")


# ── Test with a multi-step task ───────────────────────────────────────────────
# Watch Claude call 3 tools in sequence to answer one question
run_agent(
    "I'm planning a trip from London to Paris. "
    "Can you check the weather in Paris, find available flights, "
    "and tell me the GBP to EUR exchange rate? "
    "What's the population of Paris?"
)
