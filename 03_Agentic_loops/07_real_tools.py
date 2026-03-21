import os
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from anthropic import Anthropic

load_dotenv()
client = Anthropic()

# ── Real Tool 1: Web search via DuckDuckGo (no API key needed) ────────────────
def web_search(query: str) -> str:
    """Search the web and return top results. Uses DuckDuckGo's free instant answer API."""
    try:
        url = "https://api.duckduckgo.com/"
        params = {"q": query, "format": "json", "no_html": 1, "skip_disambig": 1}
        response = requests.get(url, params=params, timeout=10)
        data = response.json()

        results = []

        # Abstract (main answer if available)
        if data.get("AbstractText"):
            results.append(f"Summary: {data['AbstractText']}")
            results.append(f"Source: {data.get('AbstractURL', '')}")

        # Related topics
        for topic in data.get("RelatedTopics", [])[:5]:
            if isinstance(topic, dict) and topic.get("Text"):
                results.append(f"- {topic['Text']}")

        if results:
            return "\n".join(results)
        else:
            return f"No direct results found for '{query}'. Try a more specific query."

    except Exception as e:
        return f"Search error: {str(e)}"


# ── Real Tool 2: Fetch and extract text from a webpage ────────────────────────
def fetch_page(url: str) -> str:
    """Fetch a webpage and return its readable text content."""
    try:
        headers = {"User-Agent": "Mozilla/5.0 (research bot)"}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")

        # Remove noise
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()

        text = soup.get_text(separator="\n", strip=True)

        # Trim to avoid sending huge amounts of tokens
        lines = [l for l in text.splitlines() if len(l.strip()) > 40]
        trimmed = "\n".join(lines[:80])

        return trimmed if trimmed else "Could not extract readable content from this page."

    except requests.exceptions.Timeout:
        return "Error: Page took too long to load."
    except requests.exceptions.HTTPError as e:
        return f"Error: Page returned status {e.response.status_code}."
    except Exception as e:
        return f"Error fetching page: {str(e)}"


# ── Tool definitions ──────────────────────────────────────────────────────────
tools = [
    {
        "name": "web_search",
        "description": "Search the web for information on a topic. Returns a summary and related results. Use this first when you need to find information.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The search query. Be specific for better results."}
            },
            "required": ["query"]
        }
    },
    {
        "name": "fetch_page",
        "description": "Fetch and read the content of a specific webpage URL. Use this to get more detail from a search result.",
        "input_schema": {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "The full URL to fetch, including https://"}
            },
            "required": ["url"]
        }
    }
]


# ── Tool dispatcher with error handling ───────────────────────────────────────
def run_tool(name: str, inputs: dict) -> str:
    try:
        if name == "web_search":
            return web_search(inputs["query"])
        elif name == "fetch_page":
            return fetch_page(inputs["url"])
        else:
            return f"Unknown tool: {name}"
    except Exception as e:
        # Always return a string — never let a tool crash the loop
        return f"Tool '{name}' failed unexpectedly: {str(e)}"


# ── Agent loop ────────────────────────────────────────────────────────────────
def run_agent(question: str):
    print(f"Question: {question}\n")
    print("=" * 60)

    messages = [{"role": "user", "content": question}]
    step = 1

    while True:
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=2048,
            tools=tools,
            messages=messages
        )

        if response.stop_reason == "end_turn":
            print(f"\nFinal answer:\n{response.content[0].text}")
            break

        if response.stop_reason == "tool_use":
            messages.append({"role": "assistant", "content": response.content})

            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    print(f"Step {step}: {block.name}({block.input})")
                    result = run_tool(block.name, block.input)
                    # Show a preview so you can see what came back
                    print(f"         → {result[:120]}...\n")
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result
                    })
                    step += 1

            messages.append({"role": "user", "content": tool_results})


# ── Try it ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # run_agent("What is prompt engineering and why does it matter for businesses using AI?")
    run_agent("What is Distributed Antenna Systems and why does it matter for public venue businesses?")
