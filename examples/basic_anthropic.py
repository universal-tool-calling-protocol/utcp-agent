"""Anthropic Claude example for UtcpAgent.

This example demonstrates how to use UtcpAgent with Anthropic Claude models.
"""

import asyncio
import logging
import os
import sys
from pathlib import Path

try:
    from langchain_anthropic import ChatAnthropic
except ImportError:
    print("Error: langchain_anthropic is required for this example.")
    print("Install with: pip install langchain-anthropic")
    sys.exit(1)

from utcp_agent import UtcpAgent, UtcpAgentConfig
from dotenv import load_dotenv
from langgraph.checkpoint.memory import MemorySaver

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s:%(lineno)d - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

# Disable UTCP library logging
logging.getLogger("utcp").setLevel(logging.WARNING)

async def main():
    load_dotenv()
    """Example using Anthropic Claude models."""
    print("=== UtcpAgent with Anthropic ===")
    
    # Check for API key
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("Error: ANTHROPIC_API_KEY environment variable is required.")
        print("Set it with: export ANTHROPIC_API_KEY=your_key_here or in a .env file")
        return
    
    # Initialize Anthropic LLM
    llm = ChatAnthropic(
        model="claude-3-haiku-20240307",
        temperature=0.1,
        api_key=api_key
    )
    
    # Configure UTCP client (can load from config file or dict)
    utcp_config = {
        "load_variables_from": [
            {
                "variable_loader_type": "dotenv",
                "env_file_path": str(Path(__file__).parent / ".env")
            }
        ],
        "manual_call_templates": [
            {
                "name": "openlibrary",
                "call_template_type": "http",
                "http_method": "GET",
                "url": "https://openlibrary.org/static/openapi.json",
                "content_type": "application/json"
            }
        ]
    }
    
    # Configure agent
    agent_config = UtcpAgentConfig(
        max_tools_per_search=10,
        checkpointer=MemorySaver(),
        system_prompt="You are a helpful AI assistant with access to various tools through UTCP."
    )
    
    # Create agent
    try:
        agent = await UtcpAgent.create(
            llm=llm,
            utcp_config=utcp_config,
            agent_config=agent_config
        )
        
        # Example conversations
        print("Asking about books by a famous author...")
        response = await agent.chat("Can you find books by Isaac Asimov?")
        print(f"Agent: {response}\n")
        
        print("Asking about a specific book...")
        response = await agent.chat("Tell me about the book 'Foundation' by Isaac Asimov")
        print(f"Agent: {response}\n")
        
    except Exception as e:
        print(f"Error creating or using agent: {e}")


if __name__ == "__main__":
    asyncio.run(main())
