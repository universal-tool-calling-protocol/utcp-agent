"""UtcpAgent: A LangGraph-based agent for UTCP tool calling.

This module provides a ready-to-use agent implementation that uses LangGraph
to orchestrate UTCP tool discovery and execution. The agent follows a workflow
similar to the original example but uses LangGraph for better state management
and observability.

Workflow:
1. User question
2. Agent formulates task
3. UTCP search tool using task
4. Agent creates response (tool call or direct response)
"""

import re
import json
import logging
import uuid
from typing import Dict, Any, List, Optional, TypedDict, Annotated, Union
from dataclasses import dataclass
from pydantic import ValidationError
from langchain_core.callbacks import Callbacks
from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.base import BaseCheckpointSaver

from langgraph.graph.state import CompiledStateGraph, RunnableConfig
from utcp.exceptions import UtcpVariableNotFound
from utcp.utcp_client import UtcpClient
from utcp.data.utcp_client_config import UtcpClientConfig
from utcp.data.tool import Tool, ToolSerializer
# Configure logger
logger = logging.getLogger(__name__)


class AgentState(TypedDict):
    """State for the UTCP agent workflow."""
    messages: Annotated[List[BaseMessage], "The conversation messages"]
    current_task: str
    available_tools: List[Dict[str, Any]]  # Serialized UTCP tool dictionaries
    next_action: str  # "search_tools", "call_tool", "respond", "end"
    decision_data: Optional[Dict[str, Any]]
    iteration_count: int
    final_response: Optional[str]


@dataclass
class UtcpAgentConfig:
    """Configuration for the UTCP agent."""
    max_iterations: int = 3
    max_tools_per_search: int = 10
    system_prompt: Optional[str] = None
    checkpointer: Optional[BaseCheckpointSaver] = None
    callbacks: Optional[Callbacks] = None  # LangFuse, LangSmith, custom callbacks
    summarize_threshold: int = 80000  # Token count threshold for context summarization
    recursion_limit: int = 25


class UtcpAgent:
    """A LangGraph-based agent for UTCP tool calling.
    
    This agent uses LangGraph to orchestrate the workflow of:
    1. Analyzing user queries to formulate tasks
    2. Searching for relevant UTCP tools
    3. Deciding whether to call tools or respond directly
    4. Executing tool calls and processing results
    
    Attributes:
        llm: The language model to use for decision making
        utcp_client: The UTCP client for tool discovery and execution
        config: Agent configuration
        graph: The LangGraph StateGraph for workflow orchestration
    """
    
    def __init__(
        self,
        llm: BaseLanguageModel,
        utcp_client: UtcpClient,
        config: Optional[UtcpAgentConfig] = None,
    ):
        """Initialize the UTCP agent.
        
        Args:
            llm: LangChain language model for agent reasoning
            utcp_client: UTCP client instance for tool operations
            config: Optional agent configuration
        """
        logger.info("Initializing UtcpAgent")
        self.llm = llm
        self.utcp_client = utcp_client
        self.config = config or UtcpAgentConfig()
        
        # Set up default system prompt
        self.system_prompt = self.config.system_prompt or self._get_default_system_prompt()
        
        # Set up checkpointer before building graph
        self.checkpointer = self.config.checkpointer
        logger.info(f"Checkpointer enabled: {self.checkpointer is not None}")
        
        # Track current thread ID for conversation continuity
        self.current_thread_id: Optional[str] = None
        
        # Build the LangGraph workflow
        logger.info("Building LangGraph workflow")
        self.graph: CompiledStateGraph = self._build_graph()
        logger.info("UtcpAgent initialization complete")
    
    def _get_default_system_prompt(self) -> str:
        """Get the default system prompt for the agent."""
        return """You are a helpful AI assistant with access to a wide variety of tools through UTCP.

Your workflow:
1. When given a user query, first analyze what task needs to be accomplished
2. Search for relevant tools that can help with the task
3. Either call appropriate tools or respond directly if no tools are needed
4. Provide clear, helpful responses based on tool results or your knowledge

You have access to a special tool called 'respond' when you want to respond directly to the user without calling other tools.

Be concise and helpful in your responses."""

    def _build_graph(self) -> CompiledStateGraph:
        """Build the LangGraph workflow."""
        workflow = StateGraph(AgentState)
        
        # Define workflow nodes
        workflow.add_node("analyze_task", self._analyze_task)
        workflow.add_node("search_tools", self._search_tools)
        workflow.add_node("decide_action", self._decide_action)
        workflow.add_node("execute_tools", self._execute_tools)
        workflow.add_node("respond", self._respond)
        
        # Define workflow edges
        workflow.set_entry_point("analyze_task")
        
        workflow.add_edge("analyze_task", "search_tools")
        workflow.add_edge("search_tools", "decide_action")
        
        # Conditional routing from decide_action
        workflow.add_conditional_edges(
            "decide_action",
            self._route_decision,
            {
                "call_tool": "execute_tools",
                "respond": "respond",
                "end": END,
            }
        )
        
        workflow.add_edge("execute_tools", "analyze_task")  # Route back to decision making
        workflow.add_edge("respond", END)
        
        return workflow.compile(checkpointer=self.checkpointer)
    
    async def _analyze_task(self, state: AgentState) -> Dict[str, Any]:
        """Analyze the user query to formulate the current task."""
        messages = state["messages"]
        
        # Create task analysis prompt
        task_analysis_messages = [
            SystemMessage(content=self.system_prompt),
            SystemMessage(content="Based on the conversation history, what is the next step that needs to be accomplished? Respond with a concise next step description. Do not include 'the next step is' just the next step description."),
            *messages,
            HumanMessage(content="The next step is:\n")
        ]

        estimated_tokens = self._estimate_token_count(task_analysis_messages)
        if estimated_tokens > self.config.summarize_threshold:
            messages = await self._summarize_context(messages)
            task_analysis_messages = [
                SystemMessage(content="Based on the conversation history, what is the next step that needs to be accomplished? Respond with a concise next step description. Do not include 'the next step is' just the next step description."),
                *messages,
                HumanMessage(content="The next step is:\n")
            ]
        
        try:
            response = await self.llm.ainvoke(task_analysis_messages)
            current_task = response.content.strip()
            
            logger.info(f"[AnalyzeTask] Analyzed task: {current_task}")
            
            messages = messages + [HumanMessage(content="The next step is:\n"), AIMessage(content=current_task)]

            return {
                "current_task": current_task,
                "next_action": "search_tools",
                "messages": messages
            }
        except Exception as e:
            logger.error(f"[AnalyzeTask] Error analyzing task: {e}")
            return {
                "current_task": "Unknown task",
                "next_action": "respond",
                "messages": messages
            }
    
    async def _search_tools(self, state: AgentState) -> Dict[str, Any]:
        """Search for relevant UTCP tools based on the current task."""
        current_task = state["current_task"]
        
        try:
            # Search for relevant tools using UTCP
            logger.info(f"[SearchTools] Searching for tools for task: {current_task}")
            
            tools = await self.utcp_client.search_tools(current_task, limit=self.config.max_tools_per_search)
            
            logger.info(f"[SearchTools] Found {len(tools)} relevant tools")
            for tool in tools:
                logger.debug(f"- {tool.name}: {tool.description}")
                
            return {
                "available_tools": [ToolSerializer().to_dict(tool) for tool in tools],
                "next_action": "decide_action"
            }
            
        except Exception as e:
            logger.error(f"[SearchTools] Error searching for tools: {e}")
            return {
                "available_tools": [],
                "next_action": "decide_action"
            }
    
    async def _decide_action(self, state: AgentState) -> Dict[str, Any]:
        """Decide whether to call tools or respond directly."""
        messages = state["messages"]
        available_tools = state["available_tools"]
        current_task = state["current_task"]
        iteration_count = state.get("iteration_count", 0)
        
        # Check iteration limit
        if iteration_count >= self.config.max_iterations:
            logger.info(f"[DecideAction] Reached max iterations ({self.config.max_iterations}), responding")
            return {
                "next_action": "respond",
                "decision_data": {"action": "respond", "message": "I've reached the maximum number of iterations. Let me provide a response based on what I've gathered so far."}
            }
        
        # Increment iteration count
        new_iteration_count = iteration_count + 1
        logger.info(f"[DecideAction] Iteration {new_iteration_count}/{self.config.max_iterations}")
        
        # Prepare tool descriptions for the prompt
        if available_tools:
            tools_text = json.dumps([{
                "name": tool["name"],
                "description": tool["description"],
                "inputs": tool["inputs"]
            } for tool in available_tools], indent=2)
            logger.info(f"[DecideAction] Evaluating {len(available_tools)} available tools for decision")
        else:
            tools_text = "No tools available"
            logger.info("[DecideAction] No tools available for decision")
        
        # Create decision prompt
        decision_prompt = f"""Given the current task: "{current_task}"

Available tools:
{tools_text}

Based on the conversation and available tools, decide what to do next:
1. If you need to use a tool to accomplish the task, respond with: {{"action": "call_tool", "tool_name": "tool.name", "arguments": {{"arg1": "value1"}}}}
2. If you can answer directly without tools, respond with: {{"action": "respond", "message": "your direct response"}}
3. If the conversation is complete, respond with: {{"action": "end"}}

Respond ONLY with the JSON object, no other text."""
        
        decision_messages = [
            SystemMessage(content=self.system_prompt),
            *messages,
            HumanMessage(content=decision_prompt)
        ]

        estimated_tokens = self._estimate_token_count(decision_messages)

        if estimated_tokens > self.config.summarize_threshold:
            messages = await self._summarize_context(messages)
            decision_messages = [
                SystemMessage(content=self.system_prompt),
                *messages,
                HumanMessage(content=decision_prompt)
            ]
        
        try:
            response = await self.llm.ainvoke(decision_messages)
            decision_text = response.content.strip()
            
            # Parse the JSON response
            try:
                json_match = re.search(r'```json\n({.*?})\n```', decision_text, re.DOTALL)
                if not json_match:
                    json_match = re.search(r'({.*})', decision_text, re.DOTALL)

                decision = json.loads(json_match.group(1))
                action = decision.get("action", "respond")
                
                logger.info(f"[DecideAction] Agent decision: {action}")
                if action == "call_tool":
                    tool_name = decision.get("tool_name", "unknown")
                    logger.info(f"[DecideAction] Selected tool: {tool_name}")
                
                return {
                    "next_action": action,
                    "decision_data": decision,
                    "iteration_count": new_iteration_count,
                    "messages": messages
                }
            except json.JSONDecodeError:
                logger.warning(f"[DecideAction] Could not parse decision JSON: {decision_text}")
                return {
                    "next_action": "respond",
                    "decision_data": {"action": "respond", "message": decision_text},
                    "iteration_count": new_iteration_count,
                    "messages": messages
                }
                
        except Exception as e:
            logger.error(f"[DecideAction] Error making decision: {e}")
            return {
                "next_action": "respond",
                "decision_data": {"action": "respond", "message": f"I encountered an error: {str(e)}"},
                "iteration_count": new_iteration_count,
                "messages": messages
            }
    
    def _route_decision(self, state: AgentState) -> str:
        """Route based on the agent's decision."""
        next_action = state.get("next_action", "respond")
        logger.info(f"[RouteDecision] Routing to: {next_action}")
        return next_action
    
    async def _execute_tools(self, state: AgentState) -> Dict[str, Any]:
        """Execute the selected tool."""
        decision_data = state.get("decision_data", {})
        messages = state["messages"]
        
        tool_name = decision_data.get("tool_name")
        arguments = decision_data.get("arguments", {})
        
        if not tool_name:
            logger.error("[ExecuteTools] No tool name specified in decision")
            # Add error message to conversation
            error_msg = "Error: No tool specified"
            updated_messages = messages + [AIMessage(content=error_msg)]
            return {
                "messages": updated_messages,
                "next_action": "respond"
            }
        
        try:
            logger.info(f"[ExecuteTools] Executing tool: {tool_name} with arguments: {arguments}")
            
            # Execute the tool
            try:
                result = await self.utcp_client.call_tool(tool_name, arguments)
            except UtcpVariableNotFound as e:
                required_variables = await self.utcp_client.get_required_variables_for_registered_tool(tool_name)
                error_msg = f"Tool {tool_name} requires the following variables to be set: {required_variables}."
                logger.error("[ExecuteTools] " + error_msg)
                # Add error to messages
                updated_messages = messages + [AIMessage(content=error_msg)]
                return {
                    "messages": updated_messages,
                    "next_action": "respond"
                }

            logger.info(f"[ExecuteTools] Tool result: {str(result)[:100] + '...' if len(str(result)) > 100 else str(result)}")
            
            # Add tool call and result to messages
            tool_call_msg = AIMessage(content=f"Tool called: {tool_name} with arguments: {arguments}")
            try:
                tool_result_msg = HumanMessage.model_validate(result)
            except (ValidationError, TypeError):
                try:
                    tool_result_msg = HumanMessage.model_validate([result])
                except (ValidationError, TypeError):
                    if str(result).strip() == "":
                        tool_result_msg = HumanMessage(content="Result is empty. Try different arguments or a different tool.")
                    else:
                        tool_result_msg = HumanMessage(content=f"Tool result: {result}")
                        
                        if self._estimate_token_count([tool_result_msg]) > self.config.summarize_threshold:
                            tool_result_msg = HumanMessage(content="Result is too long to display. Try different arguments or a different tool. This is the beginning of the result: " + str(result)[:100] + "...")

            updated_messages = messages + [
                tool_call_msg,
                tool_result_msg
            ]
            
            return {
                "messages": updated_messages,
                "next_action": "analyze_task"  # Go back to decision making
            }
            
        except Exception as e:
            error_msg = f"Error executing tool {tool_name}: {str(e)}"
            logger.error("[ExecuteTools] " + error_msg)
            
            # Add error to messages
            updated_messages = messages + [AIMessage(content=error_msg)]
            return {
                "messages": updated_messages,
                "next_action": "respond"
            }
    
    async def _respond(self, state: AgentState) -> Dict[str, Any]:
        """Generate the final response to the user."""
        messages = state["messages"]
        decision_data = state.get("decision_data", {})
        current_task = state["current_task"]
        
        logger.info("[Respond] Generating response based on conversation history")
        response_prompt = f"""Based on the conversation history and the task: "{current_task}", provide a helpful summary response to the user. 

If tools were called and results obtained, summarize what was accomplished and provide the relevant information from the tool results.
If no tools were needed, provide a direct helpful response.

Be concise and helpful."""
            
        response_messages = [
            SystemMessage(content=self.system_prompt),
            *messages,
            HumanMessage(content=response_prompt)
        ]

        estimated_tokens = self._estimate_token_count(response_messages)

        if estimated_tokens > self.config.summarize_threshold:
            messages = await self._summarize_context(messages)
            response_messages = [
                SystemMessage(content=self.system_prompt),
                *messages,
                HumanMessage(content=response_prompt)
            ]
        
        try:
            response = await self.llm.ainvoke(response_messages)
            response_text = response.content.strip()
        except Exception as e:
            logger.error(f"[Respond] Error generating response: {e}")
            response_text = f"I encountered an error generating the response: {str(e)}"
        
        # Add the response to messages
        updated_messages = messages + [AIMessage(content=response_text)]
        logger.info(f"[Respond] Generated response: {response_text[:100]}{'...' if len(response_text) > 100 else ''}")
        
        return {
            "messages": updated_messages,
            "final_response": response_text
        }
    
    def _estimate_token_count(self, messages: List[BaseMessage]) -> int:
        """Estimate token count for messages (rough approximation)."""
        total_chars = sum(len(msg.content) for msg in messages)
        # Rough approximation: 1 token ≈ 4 characters for English text
        return total_chars // 4
    
    async def _summarize_context(self, messages: List[BaseMessage]) -> List[BaseMessage]:
        """Manage context size by summarizing history if needed."""
        # Keep the system message (if any) and recent messages
        system_messages = [msg for msg in messages if isinstance(msg, SystemMessage)]
        non_system_messages = [msg for msg in messages if not isinstance(msg, SystemMessage)]
        
        # Keep last few messages as-is and summarize the rest
        recent_count = 2
        if len(non_system_messages) <= recent_count:
            return messages
        
        messages_to_summarize = non_system_messages[:-recent_count]
        recent_messages = non_system_messages[-recent_count:]
        
        # Create summarization prompt
        conversation_text = "\n".join([
            f"{'Human' if isinstance(msg, HumanMessage) else 'Assistant'}: {msg.content}"
            for msg in messages_to_summarize
        ])
        
        summarization_prompt = f"""Please summarize the following conversation history concisely, preserving key information, decisions made, and context:

{conversation_text}

Provide a concise summary that captures the essential points and context."""
        
        try:
            summary_response = await self.llm.ainvoke([HumanMessage(content=summarization_prompt)])
            summary = summary_response.content.strip()
            
            # Create summarized history message
            summary_message = HumanMessage(content=f"Conversation summary: {summary}")
            
            logger.info(f"[SummarizeContext] Summarized {len(messages_to_summarize)} messages")
            
            # Return: system messages + summary + recent messages
            return system_messages + [summary_message] + recent_messages
            
        except Exception as e:
            logger.error(f"[SummarizeContext] Error summarizing history: {e}")
            # Fallback: just truncate old messages
            return system_messages + recent_messages
    
    async def chat(self, user_input: str, thread_id: Optional[str] = None) -> str:
        """Process a user input and return the agent's response.
        
        Args:
            user_input: The user's message
            thread_id: Optional thread ID for conversation continuity
            
        Returns:
            Dictionary containing:
            - "response": The agent's response text
            - "thread_id": The thread ID used for this conversation
        """
        logger.info(f"[Chat] Processing user input: {user_input[:100]}{'...' if len(user_input) > 100 else ''}")
        logger.info(f"[Chat] Thread ID: {thread_id}")
        
        try:
            # Create initial state
            initial_state = {
                "messages": [HumanMessage(content=user_input)],
                "current_task": "",
                "available_tools": [],
                "next_action": "analyze_task",
                "tool_result": None,
                "iteration_count": 0
            }
            
            # Configure for checkpointing and callbacks
            config = {}
            if self.config.recursion_limit:
                config["recursion_limit"] = self.config.recursion_limit
            if self.checkpointer:
                # Auto-generate thread_id if not provided but checkpointer is configured
                actual_thread_id = thread_id or str(uuid.uuid4())
                self.current_thread_id = actual_thread_id  # Store for conversation continuity
                config["configurable"] = {"thread_id": actual_thread_id}
                logger.info(f"[Chat] Using thread ID: {actual_thread_id}")
            if self.config.callbacks:
                config["callbacks"] = self.config.callbacks
                logger.info("[Chat] Callbacks configured")
            
            # Run the workflow
            result = await self.graph.ainvoke(initial_state, config=config)
            
            final_response = result.get("final_response", "I'm sorry, I couldn't process your request.")
            logger.info("[Chat] Workflow completed successfully.")
            
            return final_response
            
        except Exception as e:
            logger.error(f"[Chat] Error in chat: {e}")
            return f"I encountered an error: {str(e)}"
    
    async def stream(self, user_input: str, thread_id: Optional[str] = None):
        """Stream the agent's workflow execution.
        
        Args:
            user_input: The user's message
            thread_id: Optional thread ID for conversation continuity
            
        Yields:
            Workflow steps and updates
        """
        
        try:
            # Create initial state
            initial_state = {
                "messages": [HumanMessage(content=user_input)],
                "current_task": "",
                "available_tools": [],
                "next_action": "analyze_task",
                "tool_result": None,
                "iteration_count": 0
            }
            
            # Configure for checkpointing and callbacks
            config = {}
            if self.checkpointer:
                # Auto-generate thread_id if not provided but checkpointer is configured
                actual_thread_id = thread_id or str(uuid.uuid4())
                self.current_thread_id = actual_thread_id  # Store for conversation continuity
                config["configurable"] = {"thread_id": actual_thread_id}
                logger.info(f"[Chat] Using thread ID: {actual_thread_id}")
            if self.config.callbacks:
                config["callbacks"] = self.config.callbacks
                logger.info("[Chat] Callbacks configured")
            
            # Stream the workflow
            async for step in self.graph.astream(initial_state, config=config):
                yield step
                
        except Exception as e:
            logger.error(f"[Stream] Error in stream: {e}")
            yield {"error": str(e)}
    
    def get_current_thread_id(self) -> Optional[str]:
        """Get the current thread ID for conversation continuity.
        
        Returns:
            The current thread ID if available, None otherwise
        """
        return self.current_thread_id
    
    @classmethod
    async def create(
        cls,
        llm: BaseLanguageModel,
        utcp_config: Optional[Union[str, Dict[str, Any], UtcpClientConfig]] = None,
        agent_config: Optional[UtcpAgentConfig] = None,
        root_dir: Optional[str] = None,
    ) -> "UtcpAgent":
        """Create a new UtcpAgent with automatic UTCP client initialization.
        
        Args:
            llm: LangChain language model
            utcp_config: UTCP client configuration
            agent_config: Agent configuration
            root_dir: Root directory for UTCP client
            
        Returns:
            Initialized UtcpAgent instance
        """
        # Create UTCP client
        utcp_client = await UtcpClient.create(root_dir=root_dir, config=utcp_config)
        
        # Create agent
        return cls(llm=llm, utcp_client=utcp_client, config=agent_config)
