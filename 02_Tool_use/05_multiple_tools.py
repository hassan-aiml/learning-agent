import os
import json
from dotenv import load_dotenv
from anthropic import Anthropic

load_dotenv()
client = Anthropic()

# ── Multiple tools — Claude picks the right one ───────────────────────────────

def get_weather(city: str) -> str:
    data = {"london": "12°C, overcast", "new york": "22°C, sunny", "tokyo": "18°C, rain"}
    return data.get(city.lower(), f"No data for {city}")

def calculator(operation: str, a: float, b: float) -> str:
    ops = {"add": a + b, "subtract": a - b, "multiply": a * b, "divide": a / b if b != 0 else "Error: division by zero"}
    result = ops.get(operation, "Unknown operation")
    return str(result)

def get_stock_price(ticker: str) -> str:
    prices = {"AAPL": "$189.50", "GOOGL": "$141.20", "MSFT": "$378.90", "TSLA": "$245.60"}
    return prices.get(ticker.upper(), f"Ticker {ticker} not found")


# ── Tool descriptions ─────────────────────────────────────────────────────────
tools = [
    {
        "name": "get_weather",
        "description": "Get current weather for a city.",
        "input_schema": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"]
        }
    },
    {
        "name": "calculator",
        "description": "Perform basic arithmetic. Use for any math calculation.",
        "input_schema": {
            "type": "object",
            "properties": {
                "operation": {"type": "string", "enum": ["add", "subtract", "multiply", "divide"]},
                "a": {"type": "number"},
                "b": {"type": "number"}
            },
            "required": ["operation", "a", "b"]
        }
    },
    {
        "name": "get_stock_price",
        "description": "Get the current stock price for a ticker symbol.",
        "input_schema": {
            "type": "object",
            "properties": {"ticker": {"type": "string", "description": "Stock ticker e.g. AAPL"}},
            "required": ["ticker"]
        }
    }
]


# ── Tool dispatcher — maps tool names to functions ────────────────────────────
def run_tool(name: str, inputs: dict) -> str:
    if name == "get_weather":
        return get_weather(inputs["city"])
    elif name == "calculator":
        return calculator(inputs["operation"], inputs["a"], inputs["b"])
    elif name == "get_stock_price":
        return get_stock_price(inputs["ticker"])
    else:
        return f"Unknown tool: {name}"


# ── Agent loop — handles one round of tool use ────────────────────────────────
def ask_with_tools(user_message: str):
    print(f"User: {user_message}")
    messages = [{"role": "user", "content": user_message}]

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        tools=tools,
        messages=messages
    )

    # If Claude wants a tool, run it and send back the result
    if response.stop_reason == "tool_use":
        tool_block = next(b for b in response.content if b.type == "tool_use")
        print(f"  → Claude calls: {tool_block.name}({tool_block.input})")

        result = run_tool(tool_block.name, tool_block.input)
        print(f"  → Result: {result}")

        # Second call with the tool result
        messages.append({"role": "assistant", "content": response.content})
        messages.append({
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": tool_block.id, "content": result}]
        })

        final = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1024,
            tools=tools,
            messages=messages
        )
        print(f"Claude: {final.content[0].text}\n")
    else:
        print(f"Claude: {response.content[0].text}\n")


# ── Test it — Claude picks the right tool each time ──────────────────────────
ask_with_tools("What's the weather in London?")
ask_with_tools("What is 347 multiplied by 28?")
ask_with_tools("What's the current price of Apple stock?")
ask_with_tools("How are you today?")   # no tool needed — Claude handles it directly
