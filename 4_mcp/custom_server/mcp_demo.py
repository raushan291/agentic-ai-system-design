import asyncio

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_mcp_adapters.client import MultiServerMCPClient

load_dotenv()


async def run_agent():
    client = MultiServerMCPClient(
        {
            "TerminalCmdRunner": {
                "command": "python",
                "args": ["./terminal_cmd_runner_mcp.py"],
                "transport": "stdio",
            },
            "FileSystem": {
                "command": "python",
                "args": ["./filesystem_mcp.py"],
                "transport": "stdio",
            },
        }
    )

    tools = await client.get_tools()
    agent = create_agent("google_genai:gemini-2.5-flash", tools)
    response = await agent.ainvoke(
        {"messages": "list all files in the present directory."}
    )
    print(response["messages"][-1].content)
    print("---------------------------------------------------")
    response = await agent.ainvoke(
        {
            "messages": "Create a new file 'tmp.py' in the present directory with binary search code."
        }
    )
    print(response["messages"][-1].content)
    print("---------------------------------------------------")
    response = await agent.ainvoke(
        {
            "messages": "Create a directory named 'tmp_dir' and a new file 'tmp.py' in it with binary search code."
        }
    )
    print(response["messages"][-1].content)


if __name__ == "__main__":
    asyncio.run(run_agent())
