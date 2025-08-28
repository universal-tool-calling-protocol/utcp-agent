"""Basic OpenAI example for UtcpAgent.

This example demonstrates how to use UtcpAgent with OpenAI GPT models.
"""

import asyncio
import logging
import os
import sys
from pathlib import Path
try:
    from langchain_openai import ChatOpenAI
except ImportError:
    print("Error: langchain_openai is required for this example.")
    print("Install with: pip install langchain_openai")
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
    """Example using OpenAI GPT models."""
    print("=== UtcpAgent with OpenAI ===")
    
    # Check for API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable is required.")
        print("Set it with: export OPENAI_API_KEY=your_key_here or in a .env file")
        return
    
    # Initialize OpenAI LLM
    llm = ChatOpenAI(
        model="gpt-4o-mini",
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
        print("Asking about a book...")
        response = await agent.chat("Can you search for books by George Orwell?")
        print(f"Agent: {response}\n")
        
        print("Asking about a specific book...")
        response = await agent.chat("Can you find information about the book '1984'?")
        print(f"Agent: {response}\n")
        
    except Exception as e:
        print(f"Error creating or using agent: {e}")


if __name__ == "__main__":
    asyncio.run(main())
