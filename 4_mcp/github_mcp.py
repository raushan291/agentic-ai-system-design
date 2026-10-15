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
            # "github": {
            #     "command": "npx",
            #     "args": [
            #         "-y",
            #         "@modelcontextprotocol/server-github"
            #     ],
            #     "env": {
            #         "GITHUB_PERSONAL_ACCESS_TOKEN": GITHUB_TOKEN
            #     },
            #     "transport": "stdio"
            # },
            "github": {
                "transport": "http",
                "url": "https://api.githubcopilot.com/mcp/",
                "headers": {"Authorization": f"Bearer {GITHUB_TOKEN}"},
            }
        }
    )
    tools = await client.get_tools()
    agent = create_agent("google_genai:gemini-2.5-flash", tools)
    # response = await agent.ainvoke({"messages": "what are the files present in repository raushan291/test-repo? Add a new file 'tmp.py' in the raushan291/test-repo with binary search code."})
    response = await agent.ainvoke(
        {
            "messages": "remove the 'tmp.py' from the raushan291/test-repo main branch. Write commit message as 'removed tmp.py' by custom-agent using github mcp and push the changes to the repository."
        }
    )
    print(response["messages"][-1].content)


if __name__ == "__main__":
    asyncio.run(run_agent())
