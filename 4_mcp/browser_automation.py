import asyncio

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import (
    ModelCallLimitMiddleware,
    ToolCallLimitMiddleware,
)
from langchain_mcp_adapters.client import MultiServerMCPClient

load_dotenv()

# Reduce non-fatal Gemini schema warnings.
import logging

logging.getLogger("langchain_google_genai._function_utils").setLevel(logging.ERROR)


async def run_agent():
    client = MultiServerMCPClient(
        {
            "playwright": {
                "command": "npx",
                "args": ["-y", "@playwright/mcp@latest"],
                "transport": "stdio",
            }
        }
    )

    tools = await client.get_tools()
    agent = create_agent(
        "google_genai:gemini-2.5-flash",
        tools,
        system_prompt=(
            "You are a browser automation agent. "
            "Use Playwright tools to interact with webpages. "
            "First navigate to the requested URL. "
            "Use browser_snapshot to inspect the current page. "
            "Before interacting with a page, confirm the "
            "current URL and page state. "
            "If the current page is about:blank, navigate "
            "to the target URL again. "
            "Do not report success unless the tools confirm it."
        ),
        middleware=[
            ModelCallLimitMiddleware(run_limit=15, exit_behavior="end"),
            ToolCallLimitMiddleware(
                tool_name="browser_navigate", run_limit=5, exit_behavior="end"
            ),
            ToolCallLimitMiddleware(
                tool_name="browser_snapshot", run_limit=5, exit_behavior="end"
            ),
            ToolCallLimitMiddleware(
                tool_name="browser_click", run_limit=5, exit_behavior="end"
            ),
        ],
    )

    response = await agent.ainvoke(
        {
            "messages": "Go to https://google.com, search for 'Model Context Protocol', and summarize the top result"
        }
    )
    print(response["messages"][-1].content)


if __name__ == "__main__":
    asyncio.run(run_agent())
