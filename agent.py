from agents import Agent, AgentUpdatedStreamEvent, RawResponsesStreamEvent, RunItemStreamEvent, Runner, MessageOutputItem, function_tool
import httpx
from typing import Dict, Any, AsyncIterator, List, Optional, AsyncGenerator
import json
import asyncio
import os
from pydantic import BaseModel, Field
from openai.types.responses import ResponseTextDeltaEvent, ResponseFunctionCallArgumentsDeltaEvent, ResponseTextDoneEvent

class CurrencyAgent:
    def __init__(self, api_key: Optional[str] = None):
        """Initialize the Currency Agent with OpenAI-Agents SDK.
        
        Args:
            api_key: OpenAI API key (optional, can be set via OPENAI_API_KEY env var)
        """
        # Set API key if provided
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
            
        @function_tool
        async def get_exchange_rate(currency_from: str, currency_to: str, currency_date: str) -> Dict[str, Any]:
            """Get the exchange rate for a given currency pair and date.
            
            Args:
                currency_from: The currency to convert from (e.g., 'USD').
                currency_to: The currency to convert to (e.g., 'EUR').
                currency_date: The date for the exchange rate or 'latest'.
            """
            try:
                response = await httpx.AsyncClient().get(
                    f"https://api.frankfurter.app/{currency_date}",
                    params={"from": currency_from, "to": currency_to},
                )
                response.raise_for_status()

                data = response.json()
                if "rates" not in data:
                    return {"error": "Invalid API response format."}
                return data
            except httpx.HTTPError as e:
                return {"error": f"API request failed: {e}"}
            except ValueError:
                return {"error": "Invalid JSON response from API."}    
            
        # Create the agent
        self.agent = Agent(
            name="CurrencyAgent",
            instructions=(
                "You are a specialized assistant for currency conversions. "
                "Your sole purpose is to use the 'get_exchange_rate' tool to answer questions about currency exchange rates. "
                "If the user asks about anything other than currency conversion or exchange rates, "
                "politely state that you cannot help with that topic and can only assist with currency-related queries. "
                "Do not attempt to answer unrelated questions or use tools for other purposes."
            ),
            tools=[get_exchange_rate],
            model="gpt-4o-mini",  # You can change to a different model as needed
        )
        
        # Initialize runner storage
        self.runners: Dict[str, Runner] = {}
    

    def _get_or_create_runner(self, session_id: str) -> Runner:
        """Get an existing runner or create a new one for the session."""
        if session_id not in self.runners:
            self.runners[session_id] = Runner()
        return self.runners[session_id]
    
    async def invoke(self, query: str, session_id: str) -> Dict[str, Any]:
        """Asynchronous invocation of the agent."""
        runner = self._get_or_create_runner(session_id)
        
        # Complete run
        final_response = await runner.run(starting_agent=self.agent, input=query)
        
        # Format the response
        return {
            "is_task_complete": True,
            "require_user_input": False,
            "content": final_response.final_output
        }
    
    async def stream(self, query: str, session_id: str) -> AsyncGenerator[Dict[str, Any], None]:
        """Stream the agent's processing and responses."""
        runner = self._get_or_create_runner(session_id)
        
        # First yield to indicate we're starting
        yield {
            "is_task_complete": False,
            "require_user_input": False,
            "content": "Processing your currency query..."
        }
        
        # Stream the run
        run = runner.run_streamed(starting_agent=self.agent, input=query)
        
        async for event in run.stream_events():
            if isinstance(event, RawResponsesStreamEvent):

                print("\n\nDebugging raw event:", event.data)

                if isinstance(event.data, ResponseTextDoneEvent):
                    # this should be the final response from the LLM
                    #print (event.data.delta, end="", flush=True)
                    
                    yield {
                        "is_task_complete": True,
                        "require_user_input": False,
                        "content": event.data.text
                    }

                if isinstance(event.data, ResponseFunctionCallArgumentsDeltaEvent):
                    # these should be the paramters for the function tool
                    #print (event.data.delta, end="", flush=True)
                    
                    yield {
                        "is_task_complete": False,
                        "require_user_input": False,
                        "content": f"Tool call arguments: {event.data.delta}"
                    }

                elif isinstance(event.data, ResponseTextDeltaEvent):
                    # this should be the final response from the LLM
                    #print (event.data.delta, end="", flush=True)
                    
                    yield {
                        "is_task_complete": False,
                        "require_user_input": False,
                        "content": event.data.delta
                    }

            elif isinstance(event, AgentUpdatedStreamEvent):

                print("\n\nDebugging agent event:", event.new_agent.name)

                yield {
                    "is_task_complete": False,
                    "require_user_input": False,
                    "content": f"Agent started: {event.new_agent.name}"
                }

            elif isinstance(event, RunItemStreamEvent):

                print("\n\nDebugging run item event:", event.name)

                if event.name == "tool_called":
                    # this should be the tool call event
                    #print (event.item, end="", flush=True)
                    
                    yield {
                        "is_task_complete": False,
                        "require_user_input": False,
                        "content": f"Tool called: {event.item.raw_item.name} with arguments {event.item.raw_item.arguments}"
                    }

                elif event.name == "tool_output":
                    # this should be the tool output event
                    #print (event.item, end="", flush=True)
                    
                    yield {
                        "is_task_complete": False,
                        "require_user_input": False,
                        "content": f"Tool output: {event.item.output}"
                    }

    
    # For compatibility with the original implementation
    def sync_invoke(self, query: str, session_id: str) -> Dict[str, Any]:
        """Synchronous wrapper for invoke."""
        return asyncio.run(self.invoke(query, session_id))
    
    # For compatibility with the original API
    SUPPORTED_CONTENT_TYPES = ["text", "text/plain"]