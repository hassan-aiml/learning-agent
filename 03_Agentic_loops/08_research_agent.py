import os
import requests
from datetime import datetime
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from anthropic import Anthropic

load_dotenv()
client = Anthropic()

# ── System prompt — turns Claude into a focused research agent ────────────────
SYSTEM_PROMPT = """You are a professional research assistant. Your job is to research 
topics thoroughly and produce clear, structured reports.

Your process for every research request:
1. Search for the topic to get an overview
2. If you find useful URLs, fetch 1-2 of them for more detail
3. Synthesise everything into a structured report
4. Save the report using the save_report tool

Your report format must always be:
# [Topic Title]
## Summary
(2-3 sentence overview)
## Key Findings
(bullet points of the most important facts)
## Details
(more depth on the most interesting points)
## Sources
(list any URLs you found)

Be factual, concise, and professional. Do not make up information — only report what you found."""


# ── Tools ─────────────────────────────────────────────────────────────────────
def web_search(query: str) -> str:
    try:
        url = "https://api.duckduckgo.com/"
        params = {"q": query, "format": "json", "no_html": 1, "skip_disambig": 1}
        r = requests.get(url, params=params, timeout=10)
        data = r.json()
        results = []
        if data.get("AbstractText"):
            results.append(f"Summary: {data['AbstractText']}")
            results.append(f"Source: {data.get('AbstractURL', '')}")
        for topic in data.get("RelatedTopics", [])[:5]:
            if isinstance(topic, dict) and topic.get("Text"):
                results.append(f"- {topic['Text']}")
        return "\n".join(results) if results else f"No results for '{query}'."
    except Exception as e:
        return f"Search error: {str(e)}"

def fetch_page(url: str) -> str:
    try:
        headers = {"User-Agent": "Mozilla/5.0 (research bot)"}
        r = requests.get(url, headers=headers, timeout=10)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        text = soup.get_text(separator="\n", strip=True)
        lines = [l for l in text.splitlines() if len(l.strip()) > 40]
        return "\n".join(lines[:80]) or "No readable content found."
    except requests.exceptions.Timeout:
        return "Error: Page timed out."
    except requests.exceptions.HTTPError as e:
        return f"Error: HTTP {e.response.status_code}"
    except Exception as e:
        return f"Fetch error: {str(e)}"

def save_report(title: str, content: str) -> str:
    """Save the research report to a local .txt file."""
    try:
        # Create reports folder if it doesn't exist
        os.makedirs("reports", exist_ok=True)

        # Slugify the title for the filename
        safe_title = "".join(c if c.isalnum() else "_" for c in title)[:50]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"reports/{safe_title}_{timestamp}.txt"

        with open(filename, "w", encoding="utf-8") as f:
            f.write(content)

        return f"Report saved to: {filename}"
    except Exception as e:
        return f"Save error: {str(e)}"


tools = [
    {
        "name": "web_search",
        "description": "Search the web for information. Use this first to find an overview and relevant URLs.",
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"]
        }
    },
    {
        "name": "fetch_page",
        "description": "Read the full content of a webpage URL for more detail.",
        "input_schema": {
            "type": "object",
            "properties": {"url": {"type": "string"}},
            "required": ["url"]
        }
    },
    {
        "name": "save_report",
        "description": "Save the final research report to a file. Always call this as your last step.",
        "input_schema": {
            "type": "object",
            "properties": {
                "title":   {"type": "string", "description": "Short title for the report"},
                "content": {"type": "string", "description": "The full formatted report content"}
            },
            "required": ["title", "content"]
        }
    }
]

def run_tool(name: str, inputs: dict) -> str:
    try:
        if name == "web_search":  return web_search(inputs["query"])
        if name == "fetch_page":  return fetch_page(inputs["url"])
        if name == "save_report": return save_report(inputs["title"], inputs["content"])
        return f"Unknown tool: {name}"
    except Exception as e:
        return f"Tool '{name}' failed: {str(e)}"


# ── Full research agent ───────────────────────────────────────────────────────
def research(topic: str):
    print(f"\nResearching: {topic}")
    print("=" * 60)

    messages = [{"role": "user", "content": f"Research this topic and save a report: {topic}"}]
    step = 1

    while True:
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=4096,
            system=SYSTEM_PROMPT,
            tools=tools,
            messages=messages
        )

        if response.stop_reason == "end_turn":
            # Extract text if there is any in the final response
            for block in response.content:
                if hasattr(block, "text"):
                    print(f"\nAgent: {block.text}")
            print("\nDone! Check the reports/ folder for your file.")
            break

        if response.stop_reason == "tool_use":
            messages.append({"role": "assistant", "content": response.content})
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    print(f"Step {step}: {block.name}({list(block.input.keys())})")
                    result = run_tool(block.name, block.input)
                    print(f"         → {result[:100]}...")
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result
                    })
                    step += 1
            messages.append({"role": "user", "content": tool_results})


# ── CLI — type your own research question ─────────────────────────────────────
if __name__ == "__main__":
    print("Research Agent — powered by Claude")
    print("Type a topic to research, or 'quit' to exit\n")

    while True:
        topic = input("Research topic: ").strip()
        if topic.lower() == "quit":
            break
        if topic:
            research(topic)