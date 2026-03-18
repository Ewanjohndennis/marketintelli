import asyncio
import json
import sys
from mcp.client.stdio import stdio_client, StdioServerParameters
from mcp.client.session import ClientSession
from langchain_mcp_adapters.tools import load_mcp_tools


def unpack(raw):
    """Unwrap MCP TextContent list → plain Python object (dict, list, or str)."""
    if isinstance(raw, list) and raw:
        text = raw[0].get("text", "") if isinstance(raw[0], dict) else str(raw[0])
        try:
            return json.loads(text)
        except (json.JSONDecodeError, TypeError):
            return text
    return raw


def get_tool(tools, name):
    """Safely retrieve a tool by name with a helpful error if missing."""
    tool = next((t for t in tools if t.name == name), None)
    if tool is None:
        available = [t.name for t in tools]
        raise ValueError(
            f"Tool '{name}' not found on server. Available tools: {available}"
        )
    return tool


async def main():

    server_params = StdioServerParameters(
        command=sys.executable,
        args=["trends2.py"]
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:

            await session.initialize()
            tools = await load_mcp_tools(session)

            search_market   = get_tool(tools, "search_market")
            brand_frequency = get_tool(tools, "brand_frequency")
            market_graph    = get_tool(tools, "market_graph")

            keyword = input("Enter product or market: ")

            # Step 1: search — each result comes back as a TextContent item
            raw_results = await search_market.ainvoke({"keyword": keyword})
            results = []
            for item in raw_results:
                if isinstance(item, dict) and "text" in item:
                    try:
                        results.append(json.loads(item["text"]))
                    except (json.JSONDecodeError, TypeError):
                        pass
                elif isinstance(item, dict) and "title" in item:
                    results.append(item)

            # Step 2: brand frequency
            counts = unpack(await brand_frequency.ainvoke({"data": results}))
            if not isinstance(counts, dict):
                raise ValueError(f"Expected dict from brand_frequency, got: {type(counts)} — {counts}")

            # Step 3: generate and save the graph
            graph_result = unpack(await market_graph.ainvoke({"counts": counts}))

            print("\nBrand Frequency:")
            for brand, freq in sorted(counts.items(), key=lambda x: x[1], reverse=True):
                print(f"  {brand}: {freq}")
            print(f"\n{graph_result}")


asyncio.run(main())