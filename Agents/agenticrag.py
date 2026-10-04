from langchain_ollama import ChatOllama
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool
from langchain.agents import create_agent
from Agents.db_creator import create_vectore_store
from typing import Optional, Tuple

llm = ChatOllama(
    model= "qwen3:8b",
    base_url= "http://localhost:11434",
    temperature= 0.7
)

@tool
def search_mcp_knowledge_base(query: str) -> str:
    """Searches and retrives details about Model Context Protocol (MCP)"""
    retriever = create_vectore_store()
    docs = retriever.invoke(query)
    if not docs:
        return "Not context is retrived from the vector store"
    return "".join([doc.page_content for doc in docs])

@tool
def add_numbers(a: int, b: int) -> int:
    """Add two numbers and return the result."""
    return int(a) + int(b)

@tool
def subtract_numbers(a: int, b: int) -> int:
    """Subtract two numbers and return the result."""
    return int(a) - int(b)

web_search_tool = DuckDuckGoSearchRun()

tools = [search_mcp_knowledge_base, add_numbers, subtract_numbers, web_search_tool]
agents = create_agent(
    model = llm,
    tools= tools,
    system_prompt=(
        "You are an intelligent assistant equipped with specific tools.\n"
        "When using the 'search_mcp_knowledge_base' tool, ALWAYS synthesize the returned context according to the following rules:\n"
        "- Give ONLY a summary not the full detail.\n"
        "- Base your answer ONLY on the context provided by the tool.\n"
        "Use 'DuckDuckGoSearchRun' for real-time web information, and math tools for mathematical operations."
    )
)

def query_AI_agents(question: str) -> Tuple[str, Optional[str], Optional[str]]:
    result = agents.invoke({"messages": [("user", question)]})
    messages = result["messages"]
    response = messages[-1].content
    tool_name = None
    context = None
    for msg in messages:
        if getattr(msg, "name", None) == "search_mcp_knowledge_base":
            context = msg.content
            tool_name = "search_mcp_knowledge_base"
            break

    if not tool_name:
        for msg in messages:
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                tool_name = msg.tool_calls[0]["name"]
                break
            elif getattr(msg, "type", None) == "tool":
                tool_name = getattr(msg, "name", None)
                break
    return response, tool_name, context