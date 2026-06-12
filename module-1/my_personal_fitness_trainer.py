from langchain.tools import tool
from typing import Dict, Any
from tavily import TavilyClient

tavily_client = TavilyClient()

@tool
def web_search(query: str) -> Dict[str, Any]:

    """Search the web for information"""

    return tavily_client.search(query)

# %%
system_prompt = """

You are a persona trainer and professional bodybuilder. The user will give a list of muscles they want to train in a gym session.

Using the web search tool, search the web for exercises that can be use for train that muscles.

Return like 3 exercises suggestions for muscle.

"""

# %%
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver

agent = create_agent(
    model="gpt-5-nano",
    tools=[web_search],
    system_prompt=system_prompt,
    checkpointer=InMemorySaver()
)

# %%
from langchain.messages import HumanMessage

config = {"configurable": {"thread_id": "1"}}

response = agent.invoke(
    {"messages": [HumanMessage(content="I want to train my perctorals and triceps in gym")]},
    config
)

print(response['messages'][-1].content)

# %%
from pprint import pprint

pprint(response)


