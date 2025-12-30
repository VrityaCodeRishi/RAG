from dotenv import load_dotenv
from typing import Annotated, Literal, Sequence
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.tools.retriever import create_retriever_tool
from langchain_tavily import TavilySearch
from pydantic import BaseModel, Field
from langgraph.graph import END, StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

load_dotenv()

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

embeddings = OpenAIEmbeddings()
anime_vectorstore = Chroma(
    embedding_function=embeddings,
    collection_name="anubhav_anime_experiences",
    persist_directory="./chroma_db/anime"
)
game_vectorstore = Chroma(
    embedding_function=embeddings,
    collection_name="anubhav_game_experiences",
    persist_directory="./chroma_db/games"
)

anime_retriever = anime_vectorstore.as_retriever(search_kwargs={"k": 10})
game_retriever = game_vectorstore.as_retriever(search_kwargs={"k": 10})

anime_retriever_tool = create_retriever_tool(
    anime_retriever, "anime_experience",
    "Search for anime experiences including titles, ratings, years watched, favorite characters, and thoughts"
)
game_retriever_tool = create_retriever_tool(
    game_retriever, "game_experience",
    "Search for game experiences including titles, ratings, years played, platforms, playtime, and thoughts"
)

web_search = TavilySearch(max_results=3)
tools = [anime_retriever_tool, game_retriever_tool, web_search]

def is_general_question(question: str) -> bool:
    """Use LLM to determine if question is general or mentions specific entity."""
    class QuestionType(BaseModel):
        is_general: bool = Field(description="True if general question, False if mentions specific title")
        reasoning: str = Field(description="Brief explanation")
    
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    classifier = llm.with_structured_output(QuestionType)
    
    prompt = f"""Analyze if this question is general (favorites, lists, comparisons) or mentions a specific anime/game title.

General: favorites, best, highest rated, lists, "what did you watch/play"
Specific: mentions a specific title like "Attack on Titan" or "God of War"

Question: {question}"""
    
    result = classifier.invoke(prompt)
    return result.is_general

def get_retriever_results(messages):
    """Extract retriever tool results from messages."""
    results = {}
    for msg in messages:
        if msg.type == "tool" and msg.name in ["anime_experience", "game_experience"]:
            content = msg.content if hasattr(msg, "content") else str(msg)
            if content and len(content.strip()) >= 50 and "no results" not in content.lower():
                results[msg.name] = content
    return results

def agent(state: AgentState):
    """Main agent that routes questions to appropriate tools."""
    question = state["messages"][0].content
    model = ChatOpenAI(model="gpt-4o", temperature=0)
    
    if is_general_question(question):
        model = model.bind_tools([anime_retriever_tool, game_retriever_tool])
        system_msg = """You are Anubhav. The user asks a general question.
        Use BOTH anime_experience and game_experience tools with the complete question as query."""
    else:
        model = model.bind_tools(tools)
        system_msg = """You are Anubhav. When asked about anime/games:
        1. Use web search to identify if it's an anime or game
        2. Use anime_experience or game_experience to find your experiences
        3. Answer based on your stored experiences"""
    
    response = model.invoke([HumanMessage(content=system_msg)] + state["messages"])
    return {"messages": [response]}

def classify_entity(state: AgentState):
    """Classify entity type from web search results."""
    class EntityType(BaseModel):
        entity_type: Literal["anime", "game", "both", "unknown"] = Field(description="Entity type")
        entity_name: str = Field(description="Entity name")
    
    messages = state["messages"]
    question = messages[0].content
    
    search_results = ""
    for msg in reversed(messages):
        if msg.type == "tool":
            search_results = msg.content
            break
    
    if not search_results:
        return {"messages": [AIMessage(content="unknown")]}
    
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    classifier = llm.with_structured_output(EntityType)
    
    entity_name = question.split("about")[-1].split("?")[0].strip() if "about" in question else question.split("?")[0].strip()
    
    result = classifier.invoke(f"""Based on web search, determine if "{entity_name}" is an anime, game, or both.

Question: {question}
Search Results: {search_results[:1000]}

Return "unknown" if cannot determine.""")
    
    if result.entity_type == "unknown":
        return {"messages": [AIMessage(content="unknown")]}
    
    model = ChatOpenAI(model="gpt-4o", temperature=0)
    if result.entity_type == "anime":
        model = model.bind_tools([anime_retriever_tool])
        system_msg = f"You must use the anime_experience tool to search for information about '{result.entity_name}'. Always call the tool."
    elif result.entity_type == "game":
        model = model.bind_tools([game_retriever_tool])
        system_msg = f"You must use the game_experience tool to search for information about '{result.entity_name}'. Always call the tool."
    else:
        model = model.bind_tools([anime_retriever_tool, game_retriever_tool])
        system_msg = f"You must use both anime_experience and game_experience tools to search for information about '{result.entity_name}'. Always call the tools."
    
    response = model.invoke([HumanMessage(content=system_msg), HumanMessage(content=f"User Question: {question}")])
    return {"messages": [response]}

def check_results(state: AgentState) -> Literal["generate", "not_found"]:
    """Check if retriever tools found meaningful results."""
    results = get_retriever_results(state["messages"])
    
    if not results:
        return "not_found"

    return "generate"

def generate(state: AgentState):
    """Generate answer from retrieved context."""
    messages = state["messages"]
    question = messages[0].content
    results = get_retriever_results(messages)
    
    if not results:
        return {"messages": [AIMessage(content="I don't have enough information in my experiences to answer that question.")]}
    
    context = "\n".join(results.values())
    
    system_prompt = """You are Anubhav. Answer based on the context provided.

IMPORTANT:
- For year questions (e.g., "What anime in 2020?"), filter by "Year Watched" or "Year Played"
- For list questions, provide all matching items
- For favorites/highest rated, identify highest rating from context
- Only use information from context
- If insufficient info, say you don't have enough information"""
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "Context: {context}\n\nQuestion: {question}")
    ])
    
    chain = prompt | ChatOpenAI(model="gpt-4o", temperature=0) | StrOutputParser()
    response = chain.invoke({"context": context, "question": question})
    return {"messages": [AIMessage(content=response)]}

def not_found(state: AgentState):
    """Handle case when no results found."""
    messages = state["messages"]
    anime_searched = any(msg.type == "tool" and msg.name == "anime_experience" for msg in messages)
    game_searched = any(msg.type == "tool" and msg.name == "game_experience" for msg in messages)
    
    if anime_searched and game_searched:
        response = "I don't have enough information in my experiences to answer that question."
    elif anime_searched:
        response = "I don't have any personal experience or information about that in my anime database."
    elif game_searched:
        response = "I don't have any personal experience or information about that in my game database."
    else:
        response = "I don't have enough information in my experiences to answer that question."
    
    return {"messages": [AIMessage(content=response)]}

def route_after_tools(state: AgentState) -> Literal["classify_entity", "generate", "not_found"]:
    """Route after tool execution."""
    messages = state["messages"]
    last_msg = messages[-1]
    
    if last_msg.type == "tool" and hasattr(last_msg, "name") and "tavily" in last_msg.name.lower():
        return "classify_entity"
    
    if any(msg.type == "tool" and msg.name in ["anime_experience", "game_experience"] for msg in messages):
        return check_results(state)
    
    return "not_found"

def route_after_classify(state: AgentState) -> Literal["tools", "not_found"]:
    """Route after classification."""
    messages = state["messages"]
    last_msg = messages[-1]
    
    if isinstance(last_msg, AIMessage) and "unknown" in last_msg.content.lower():
        return "not_found"
    
    if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
        return "tools"
    
    return "not_found"

graph = StateGraph(AgentState)
graph.add_node("agent", agent)
graph.add_node("tools", ToolNode(tools))
graph.add_node("classify_entity", classify_entity)
graph.add_node("generate", generate)
graph.add_node("not_found", not_found)

graph.add_edge(START, "agent")
graph.add_conditional_edges("agent", tools_condition, {"tools": "tools", END: END})
graph.add_conditional_edges("tools", route_after_tools, {
    "classify_entity": "classify_entity",
    "generate": "generate",
    "not_found": "not_found"
})
graph.add_conditional_edges("classify_entity", route_after_classify, {
    "tools": "tools",
    "not_found": "not_found"
})
graph.add_edge("generate", END)
graph.add_edge("not_found", END)

graph = graph.compile()

if __name__ == "__main__":
    pass
