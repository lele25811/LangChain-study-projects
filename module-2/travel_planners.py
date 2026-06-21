# %% [markdown]
# ## Travel Planner Multi Agents

# %% [markdown]
# ### Tool setup

# %%
print("ciao")

# %%
import asyncio

from langchain_mcp_adapters.client import MultiServerMCPClient
from mcp.shared.exceptions import McpError
from mcp.types import CallToolResult, TextContent

RETRYABLE_MCP_CODES = {-32603}

class RetryMCPInterceptor:
    """Intercept MCP tool calls: retry transient failures, surface all errors gracefully.

    - Retryable McpError codes (e.g. -32603): retry with exponential backoff.
    - Non-retryable McpError codes (e.g. -32602): return error message immediately.
    - Any other exception (fetch failed, network errors, etc.): retry then return error message.
    """

    def __init__(self, max_retries: int = 3):
        self.max_retries = max_retries

    async def __call__(self, request, handler):
        last_error = None
        for attempt in range(self.max_retries):
            try:
                return await handler(request)
            except McpError as exc:
                last_error = exc
                print(f"[MCP interceptor] {type(exc).__name__} on {request.name} "
                      f"(code {exc.error.code}, attempt {attempt+1}/{self.max_retries}): {exc}")
                if exc.error.code not in RETRYABLE_MCP_CODES:
                    return CallToolResult(
                        content=[TextContent(type="text", text=f"Tool call failed (non-retryable): {exc}")],
                        isError=False,
                    )
            except Exception as exc:
                last_error = exc
                print(f"[MCP interceptor] {type(exc).__name__} on {request.name} "
                      f"(attempt {attempt+1}/{self.max_retries}): {exc}")

            if attempt < self.max_retries - 1:
                await asyncio.sleep(2 ** attempt)

        print(f"[MCP interceptor] all {self.max_retries} retries exhausted for {request.name}")
        return CallToolResult(
            content=[TextContent(type="text", text=f"Tool call failed after {self.max_retries} attempts: {last_error}")],
            isError=False,
        )

client = MultiServerMCPClient(
    {
        "travel_server": {
                "transport": "streamable_http",
                "url": "https://mcp.kiwi.com"
            }
    },
    tool_interceptors=[RetryMCPInterceptor()],
)

tools = await client.get_tools()

# %%
from typing import Dict, Any
from tavily import TavilyClient
from langchain.tools import tool

tavily_client = TavilyClient()

@tool
def web_search(query: str, search_number: int, max_search_number: int) -> Dict[str, Any]:
    """Search the web for information. You must track your search count by providing
    search_number (starting at 1) and max_search_number on every call.
    Queries must use only plain text characters. Do not use accented or special characters     
      (e.g., use 'capacite' instead of 'capacité').
    """
    if search_number > max_search_number:
        return {"message": "Search limit reached. Please summarize your findings and provide your final answer."}
    try:
        return tavily_client.search(query)
    except Exception as e:
        return {"error": str(e)}

# %%
from langchain_community.utilities import SQLDatabase

db = SQLDatabase.from_uri("sqlite:///resources/Chinook.db")

@tool
def query_playlist_db(query: str) -> str:

    """Query the database for playlist information"""

    try:
        return db.run(query)
    except Exception as e:
        return f"Error querying database: {e}"

# %% [markdown]
# ### Create State

# %%
from langchain.agents import AgentState

class TravelState(AgentState):
    destination: str
    start_date: str
    end_date: str

    budget_flight: float
    budget_hotel: float
    budget_activities: float

    flights: dict
    hotels: dict
    activities: dict

# %% [markdown]
# ### Create Subagents

# %%
from langchain.agents import create_agent

# Flight agents
flight_agent = create_agent(
    model="gpt-5-nano",
    tools=tools,
    system_prompt="""
    You are a Flight Search Specialist.

    Your only responsibility is finding the best flight options for the user's trip.

    You will receive:
    - Origin city
    - Destination city
    - Travel dates
    - Flight budget

    Your tasks:
    1. Find flight options that satisfy the user's requirements.
    2. Prioritize flights that fit within the allocated flight budget.
    3. Consider convenience, duration, number of layovers, and overall value.
    4. Provide a concise recommendation with reasoning.

    You must NOT:
    - Recommend hotels.
    - Recommend activities.
    - Create a complete travel itinerary.
    - Modify the user's budget.

    Return only flight-related recommendations and relevant flight information.
    """
)

# %%
# hotel_agent
hotel_agent = create_agent(
    model="gpt-5-nano",
    tools=[web_search],
    system_prompt="""
    You are a Hotel Search Specialist.

    Your only responsibility is finding the most suitable accommodation for the trip.

    You will receive:
    - Destination city
    - Travel dates
    - Hotel budget

    Your tasks:
    1. Find accommodation options that fit the allocated hotel budget.
    2. Consider location, quality, amenities, and value.
    4. Explain why the recommended options are suitable.

    You must NOT:
    - Search for flights.
    - Recommend activities.
    - Build a full travel plan.
    - Modify the user's budget.

    Return only hotel-related recommendations and relevant accommodation information.
    """
)

# %%
# Activity agent
activity_agent = create_agent(
    model="gpt-5-nano",
    tools=[web_search],
    system_prompt="""
    You are a Travel Activities Specialist.

    Your only responsibility is suggesting activities and attractions for the trip.

    You will receive:
    - Destination city
    - Travel dates
    - Activity budget

    Your tasks:
    1. Recommend activities, attractions, and experiences that match the user's interests.
    2. Stay within the allocated activities budget.
    3. Balance cultural, recreational, and local experiences when appropriate.
    4. Provide a short explanation for each recommendation.

    You must NOT:
    - Search for flights.
    - Recommend hotels.
    - Build a complete travel itinerary.
    - Modify the user's budget.

    Return only activity-related recommendations and relevant travel experiences.
    """
)


# %% [markdown]
# ## Main Coordinator

# %%
from langchain.tools import ToolRuntime
from langchain.messages import HumanMessage, ToolMessage
from langgraph.types import Command

@tool
async def search_flights(runtime: ToolRuntime) -> str:
    """ Travel agent searches for flights to the desired destination"""
    # Usiamo .get() per evitare il KeyError
    origin = runtime.state.get("origin")
    destination = runtime.state.get("destination")
    start_date = runtime.state.get("start_date")
    end_date = runtime.state.get("end_date")
    budget = runtime.state.get("budget_flight")
    
    # Controllo di sicurezza se mancano i dati fondamentali
    if not origin or not destination:
        return "Error: Missing origin or destination in state. Please call 'update_state' first with the extracted trip details."

    response = await flight_agent.ainvoke({"messages": [HumanMessage(content=f"Find flights from {origin} to {destination} between the {start_date} and {end_date} with {budget} euro of budget")]})
    return response["messages"][-1].content

@tool
async def search_hotels(runtime: ToolRuntime) -> str:
    """ Hotel agent searches for best hotels for the destination and budget and days"""
    destination = runtime.state.get("destination")
    start_date = runtime.state.get("start_date")
    end_date = runtime.state.get("end_date")
    budget = runtime.state.get("budget_hotel")
    
    if not destination:
        return "Error: Missing destination in state. Please call 'update_state' first."

    response = hotel_agent.invoke({"messages": [HumanMessage(content=f"Find the best hotels in {destination} between the {start_date} and {end_date} with {budget} euro of budget")]})
    return response['messages'][-1].content

@tool
async def search_activities(runtime: ToolRuntime) -> str:
    """ Activity agent searches for best activity for the destination and budget and days"""
    destination = runtime.state.get("destination")
    start_date = runtime.state.get("start_date")
    end_date = runtime.state.get("end_date")
    budget = runtime.state.get("budget_activities")
    
    if not destination:
        return "Error: Missing destination in state. Please call 'update_state' first."

    response = activity_agent.invoke({"messages": [HumanMessage(content=f"Find the best activity in {destination} between the {start_date} and {end_date} with {budget} euro of budget")]})
    return response['messages'][-1].content

@tool
def update_state(
    destination: str,
    start_date: str,
    end_date: str,
    budget_flight: float,
    budget_hotel: float,
    budget_activities: float,
    runtime: ToolRuntime
) -> str:
    """Update the state when you know all of the values: origin, destination, budgets. 
    This tool must be called alone, without any other tool calls. It must complete and return to make,
    the information available to other tools."""

    return Command(update={
        "destination": destination,
        "start_date": start_date,
        "end_date": end_date,
        "budget_flight": budget_flight,
        "budget_hotel": budget_hotel,
        "budget_activities": budget_activities,
        "messages": [
            ToolMessage(
                "Successfully updated state",
                tool_call_id=runtime.tool_call_id
            )
        ]
    })

# %%
from langchain.agents import create_agent

coordinator = create_agent(
    model="gpt-5-nano",
    tools=[search_flights, search_hotels, search_activities, update_state],
    state_schema=TravelState,
    system_prompt="""
    You are the Travel Planner Supervisor.
    Your role is to coordinate specialized travel agents and produce a complete travel plan.

    CRITICAL RULE: 
    - NEVER ASK QUESTIONS TO THE USER. 
    - DO NOT ASK FOR CONFIRMATION OR PREFERENCES.
    - If the user provides a total budget, you MUST immediately divide it yourself using reasonable assumptions (e.g., 40% flights, 40% hotel, 20% activities) and call the 'update_state' tool as your very first action.
    - If any preference is missing, make a standard choice (Economy flights, 3-star/mid-range hotels, standard tourist activities) and proceed.

    Steps to follow:
    1. Look at the user's input (dates, budget, destination, origin).
    2. Automatically calculate and allocate the budget fields.
    3. Call 'update_state' to store these details.
    4. Call 'search_flights', 'search_hotels', and 'search_activities' using the specialists.
    5. Construct the final plan.

    Your final answer must always contain:
    - Recommended flights
    - Recommended hotel
    - Recommended activities
    - Estimated total cost
    - Budget breakdown
    """
)

# %% [markdown]
# ## Test

# %%
from langchain.messages import HumanMessage

response = await coordinator.ainvoke(
    {
        "messages": [HumanMessage(content="I'm from Rome and I'd like a travel in Los Angeles from 07/02/2027 to 15/02/2027, with 2000,00 euro of budgets only for one traveler")],
    },
    config={"tags": ["WP"], "recursion_limit": 40},  #tag traces to make them easy to find in Langsmith. Increase number of steps the agent can take to 40.
)

# %%
from pprint import pprint

pprint(response)

# %%
print(response["messages"][-1].content)


