import os
import asyncio
from dotenv import load_dotenv

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.messages import HumanMessage
from langchain.agents import create_agent
from langgraph_supervisor import create_supervisor
from mcp.client.stdio import stdio_client, StdioServerParameters
from mcp.client.session import ClientSession
from langchain_mcp_adapters.tools import load_mcp_tools

load_dotenv()

hf = os.getenv("HF_TOKEN")

endpoint = HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-20b",
    max_new_tokens=800,
    temperature=0.5
)

llm = ChatHuggingFace(llm=endpoint)


async def main():

    server_params = StdioServerParameters(
        command="python",
        args=["mcp_server.py"]
    )

    async with stdio_client(server_params) as (read, write):

        async with ClientSession(read, write) as session:

            tools = await load_mcp_tools(session)

        news_agent = create_agent(
            model=llm,
            tools=tools,
            name="News Agent",
            system_prompt="""
You retrieve the latest news articles about companies.

Always call get_news before answering.
Return only the news.
"""
        )

        analysis_agent = create_agent(
            model=llm,
            tools=tools,
            name="Analysis Agent",
            system_prompt="""
Use analyze_market to produce the final market insight paragraph.
"""
        )

        supervisor = create_supervisor(
            model=llm,
            agents=[news_agent, analysis_agent],
            name="Supervisor",
            system_prompt="""
Step 1: Call News Agent to collect news
Step 2: Send news to Analysis Agent
Step 3: Return the insight
"""
        )

        app = supervisor.compile()

        topic = input("Enter company or industry: ")

        query = f"Get latest news and market insight about {topic}"

        response = await app.ainvoke({
            "messages": [HumanMessage(content=query)]
        })

        print("\nMarket Insight:\n")
        print(response["messages"][-1].content)


asyncio.run(main())