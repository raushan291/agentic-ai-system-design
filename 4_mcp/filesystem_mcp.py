from dotenv import load_dotenv

load_dotenv()
import asyncio
import os

from langchain.agents import create_agent
from langchain_mcp_adapters.client import MultiServerMCPClient

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")


async def run_agent():
    client = MultiServerMCPClient(
        {
            "filesystem": {
                "command": "npx",
                "args": ["-y", "@modelcontextprotocol/server-filesystem", "."],
                "transport": "stdio",
            }
        }
    )

    tools = await client.get_tools()
    agent = create_agent("google_genai:gemini-2.5-flash", tools)
    response = await agent.ainvoke(
        {
            "messages": "Create a new file 'tmp.py' in /home/raushan/Documents/code/Projects/agentic-ai-system-design/mcp with binary search code."
        }
    )
    print(response["messages"][-1].content)


if __name__ == "__main__":
    asyncio.run(run_agent())
