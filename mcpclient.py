import asyncio
from fastmcp import Client

async def main():
    client = Client("http://localhost:8000/mcp")

    async with client:
        tools = await client.list_tools()
        print("Tools:", tools)

        result = await client.call_tool(
            "search_wikipedia",
            {"query": "graph rag"}
        )
        print("Result:", result)

if __name__ == "__main__":
    asyncio.run(main())
