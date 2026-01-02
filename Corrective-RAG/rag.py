import os
from dotenv import load_dotenv
from typing import List, TypedDict
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.graph import StateGraph, START, END
from langchain_core.documents import Document

load_dotenv()

embeddings = OpenAIEmbeddings()

vectorstore = Chroma(
    embedding_function=embeddings,
    persist_directory="./chroma_db/recipes",
    collection_name="recipe_assistant",
)

retriever = vectorstore.as_retriever()

class GradeDocument(BaseModel):
    """Binary score for relevance check on retrieved documents"""
    binary_score: bool = Field(description="True if the document is relevant to the query, False otherwise")


llm = ChatOpenAI(model="gpt-5.2", temperature=0)
structured_llm_grader = llm.with_structured_output(GradeDocument)

grade_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a grader assessing relevance of retrieved recipe documents to a user question.
    If the document contains recipes, cooking techniques, ingredients, or dietary information related to the question, grade it as relevant.
    Give a binary score 'yes' or 'no' to indicate whether the document is relevant to the question."""),
    ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
])

retrieval_grader = grade_prompt | structured_llm_grader

prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise."""),
    ("human", "Question: {question}\n\nContext: {context}\n\nAnswer:"),
])

def format_docs(docs) -> str:
    """Format a list of documents into a string for retrieval"""
    return "\n\n".join([doc.page_content for doc in docs])

rag_chain = prompt | llm | StrOutputParser()

system = """You are a question re-writer that converts an input question to a better version optimized for web search.
Look at the input and reason about the underlying semantic intent. For recipe questions, include relevant keywords like 
recipe names, ingredients, dietary restrictions, or cooking techniques."""

re_write_prompt = ChatPromptTemplate.from_messages([
    ("system", system),
    ("human", "Here is the initial question: \n\n {question} \n Formulate an improved question."),
])

question_rewriter = re_write_prompt | llm | StrOutputParser()

web_search_tool = TavilySearchResults(k=3)

class GraphState(TypedDict):
    question: str
    generation: str
    web_search: str
    documents: List


def retrieve(state):
    """Retrieve relevant documents from the vector database"""
    print("\n" + "="*60)
    print("NODE: RETRIEVE")
    print("="*60)
    question = state["question"]
    print(f"Question: {question}")
    documents = retriever.invoke(question)
    print(f"Retrieved {len(documents)} documents from vector database")
    return {"documents": documents,"question": question}

def generate(state):
    """Generate a response using the retrieved documents"""
    print("\n" + "="*60)
    print("NODE: GENERATE")
    print("="*60)
    documents = state["documents"]
    question = state["question"]
    print(f"Question: {question}")
    print(f"Using {len(documents)} document(s) for generation")
    formatted_context = format_docs(documents)
    print("Generating answer with LLM...")
    generation = rag_chain.invoke({"question": question, "context": formatted_context})
    print("Answer generated successfully")
    return {"generation": generation,"question": question,"documents": documents}

def grade_documents(state):
    """ Check the relevance of the retrieved documents"""
    print("\n" + "="*60)
    print("NODE: GRADE DOCUMENTS")
    print("="*60)
    documents = state["documents"]
    question = state["question"]
    print(f"Question: {question}")
    print(f"Grading {len(documents)} document(s)...")
    filtered_docs = []
    web_search = "No"
    if len(documents) == 0:
        print("No documents retrieved from vector database")
        web_search = "Yes"
    else:
        for i, d in enumerate(documents, 1):
            print(f"  Checking document {i}/{len(documents)}...", end=" ")
            grade = retrieval_grader.invoke({"document": d.page_content, "question": question})
            if grade.binary_score:
                print("RELEVANT")
                filtered_docs.append(d)
            else:
                print("NOT RELEVANT")
                web_search = "Yes"
    
    print(f"\nResults: {len(filtered_docs)} relevant, {len(documents) - len(filtered_docs)} not relevant")
    print(f"Web search needed: {web_search}")
    return {"documents": filtered_docs,"web_search": web_search,"question": question}

def transform_query(state):
    """ Transform the question to a better version optimized for web search"""
    print("\n" + "="*60)
    print("NODE: TRANSFORM QUERY")
    print("="*60)
    question = state["question"]
    print(f"Original question: {question}")
    print("Rewriting question for web search...")
    rewritten_question = question_rewriter.invoke({"question": question})
    print(f"Rewritten question: {rewritten_question}")
    return {"question": rewritten_question}

def web_search(state):
    """ Search the web for relevant information"""
    print("\n" + "="*60)
    print("NODE: WEB SEARCH")
    print("="*60)
    question = state["question"]
    documents = state.get("documents", [])
    print(f"Search query: {question}")
    print("Searching web with Tavily...")
    docs = web_search_tool.invoke({"query": question})
    print(f"Found {len(docs)} web search results")
    web_results = "\n".join([d["content"] for d in docs])
    web_results = Document(page_content=web_results)
    documents.append(web_results)
    print(f"Total documents now: {len(documents)} (including web results)")
    return {"documents": documents,"question": question}

def decide_to_generate(state):
    """Decide whether to generate or transform query."""
    print("\n" + "="*60)
    print("NODE: DECIDE TO GENERATE")
    print("="*60)
    web_search = state["web_search"]
    
    if web_search == "Yes":
        print("DECISION: Documents not fully relevant")
        print("   Routing to: TRANSFORM QUERY -> WEB SEARCH -> GENERATE")
        return "transform_query"
    else:
        print("DECISION: All documents are relevant")
        print("   Routing to: GENERATE (direct)")
        return "generate"

graph = StateGraph(GraphState)

graph.add_node("retrieve", retrieve)
graph.add_node("generate", generate)
graph.add_node("grade_documents", grade_documents)
graph.add_node("transform_query", transform_query)
graph.add_node("web_search", web_search)

graph.add_edge(START, "retrieve")
graph.add_edge("retrieve", "grade_documents")
graph.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {
        "transform_query": "transform_query",
        "generate": "generate"
    }
)
graph.add_edge("transform_query", "web_search")
graph.add_edge("web_search", "generate")
graph.add_edge("generate", END)

builder = graph.compile()

if __name__ == "__main__":
    builder.get_graph().draw_mermaid_png()
    result = builder.invoke({"question": "How to make chocolate cake?"})
    print("FINAL ANSWER:")
    print(result["generation"])
