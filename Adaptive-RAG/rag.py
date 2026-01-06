
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.tools.tavily_search import TavilySearchResults
from pydantic import BaseModel, Field
from typing import Literal
from typing_extensions import TypedDict
from typing import List
from langgraph.graph import StateGraph, START, END

load_dotenv()

class RouteQuery(BaseModel):
    """The LLM will return this structure when routing"""
    datasource: Literal["official_docs", "community"] = Field(
        description="Choose 'official_docs' for API/documentation questions, 'community' for errors/troubleshooting"
    )
    reasoning: str = Field(
        description="Explain why you chose this route"
    )

llm = ChatOpenAI(model="gpt-5.2", temperature=0)
route_llm = llm.with_structured_output(RouteQuery)

route_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a router for technical questions.
    
    Route to 'official_docs' if:
    - Question asks about API documentation
    - Question asks "how to use" a function/class
    - Question asks about syntax or parameters
    
    Route to 'community' if:
    - Question asks about errors or troubleshooting
    - Question asks "how to fix" something
    - Question mentions specific error messages"""),
    ("human", "{question}")
])

question_router = route_prompt | route_llm

web_search = TavilySearchResults(max_results=3)

def search_official_docs(question: str) -> str:
    query = f"{question} site:docs.python.org OR site:pandas.pydata.org OR site:react.dev"
    results = web_search.invoke({"query": query})
    return results

def search_community(question: str):
    """Search community solutions (Stack Overflow, GitHub)"""
    query = f"{question} stackoverflow OR github"
    results = web_search.invoke({"query": query})
    return results

class GradeDocument(BaseModel):
    """LLM evaluates if documents are relevant"""
    binary_score: str = Field(
        description="yes if the document is relevant, no if not"
    )

grader_llm = llm.with_structured_output(GradeDocument)


grade_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a grader. Check if the search results are relevant to the question.
    If they contain useful information to answer the question, say 'yes'.
    If they're completely off-topic, say 'no'."""),
    ("human", "Question: {question}\n\nSearch Results: {documents}")
])

document_grader = grade_prompt | grader_llm

def generate_answer(question: str, search_results: str) -> str:
    """Generate answer from search results"""
    context = "\n\n".join([result.get("content", "") for result in search_results])

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful technical assistant. Answer the question using ONLY the provided search results.
        If the search results don't contain the answer, say so.
        Always provide code examples if relevant."""),
        ("human", "Question: {question}\n\nContext: {context}\n\nAnswer:")
    ])

    chain = prompt | llm
    answer = chain.invoke({"question": question, "context": context})
    return answer.content

class GradeHallucinations(BaseModel):
    """LLM evaluates if the answer contains hallucinations"""
    binary_score: str = Field(
        description="yes if the answer contains hallucinations, no if not"
    )

hallucination_llm = llm.with_structured_output(GradeHallucinations)

hallucination_prompt = ChatPromptTemplate.from_messages([
    ("system", """Check if the answer is grounded in the search results.
    'yes' = all claims can be verified in the search results
    'no' = answer contains information not in the search results"""),
    ("human", "Search Results: {documents}\n\nAnswer: {generation}")
])

hallucination_grader = hallucination_prompt | hallucination_llm

class GradeAnswer(BaseModel):
    """LLM evaluates if the answer is correct"""
    binary_score: str = Field(
        description="yes if the answer is correct, no if not"
    )

answer_quality_llm = llm.with_structured_output(GradeAnswer)

answer_quality_prompt = ChatPromptTemplate.from_messages([
    ("system", """Check if the answer fully addresses the user's question.
    'yes' = answer completely solves the problem
    'no' = answer is incomplete or doesn't address the question"""),
    ("human", "Question: {question}\n\nAnswer: {generation}")
])

answer_quality_grader = answer_quality_prompt | answer_quality_llm

class GraphState(TypedDict):
    """Shared state that flows through all nodes"""
    question: str
    route_decision: str
    documents: List
    filtered_documents: List
    generation: str

def route_question(state: GraphState):
    """Route decision using your existing router"""
    print("NODE: ROUTE QUESTION")    
    question = state["question"]
    route = question_router.invoke({"question": question})
    
    print(f"Decision: {route.datasource}")
    print(f"Reasoning: {route.reasoning}")
    
    return {"route_decision": route.datasource}


def search_official_docs_node(state: GraphState):
    """Uses your existing search_official_docs()"""
    print("NODE: SEARCH OFFICIAL DOCS")
    
    question = state["question"]
    results = search_official_docs(question)
    print(f"Found {len(results)} results")
    
    return {"documents": results}

def search_community_node(state: GraphState):
    """Uses your existing search_community()"""
    print("NODE: SEARCH COMMUNITY")
    
    question = state["question"]
    results = search_community(question)
    print(f"Found {len(results)} results")
    
    return {"documents": results}

def grade_documents(state: GraphState):
    """Uses your existing document_grader"""
    print("NODE: GRADE DOCUMENTS")
    
    question = state["question"]
    documents = state["documents"]
    
    filtered = []
    for i, doc in enumerate(documents, 1):
        content = doc.get("content", "")
        grade = document_grader.invoke({
            "question": question,
            "documents": content
        })
        
        if grade.binary_score == "yes":
            print(f"  Doc {i}: RELEVANT")
            filtered.append(doc)
        else:
            print(f"  Doc {i}: NOT RELEVANT")
    
    print(f"Result: {len(filtered)}/{len(documents)} kept")
    return {"filtered_documents": filtered}

def generate(state: GraphState):
    """Uses your existing generate_answer()"""
    print("NODE: GENERATE ANSWER")
    
    question = state["question"]
    filtered_docs = state["filtered_documents"]
    
    if not filtered_docs:
        return {"generation": "Sorry, no relevant information found."}
    
    answer = generate_answer(question, filtered_docs)
    print("Answer generated")
    
    return {"generation": answer}

def grade_hallucination(state: GraphState):
    """Uses your existing hallucination_grader"""
    print("NODE: GRADE HALLUCINATION")
    
    filtered_docs = state["filtered_documents"]
    generation = state["generation"]
    
    context = "\n\n".join([doc.get("content", "") for doc in filtered_docs])
    grade = hallucination_grader.invoke({
        "documents": context,
        "generation": generation
    })
    
    if grade.binary_score == "yes":
        print("Answer is grounded")
    else:
        print("Contains unverified info")
    
    return {}

def grade_answer_quality(state: GraphState):
    """Uses your existing answer_quality_grader"""
    print("NODE: GRADE ANSWER QUALITY")
    
    question = state["question"]
    generation = state["generation"]
    
    grade = answer_quality_grader.invoke({
        "question": question,
        "generation": generation
    })
    
    if grade.binary_score == "yes":
        print("Answer addresses question")
    else:
        print("Answer may be incomplete")
    
    return {}

def decide_search_route(state: GraphState) -> str:
    """Decides which search node to use based on route_decision"""
    return state["route_decision"]

def build_graph():
    """Create the LangGraph workflow"""
    workflow = StateGraph(GraphState)
    

    workflow.add_node("route", route_question)
    workflow.add_node("search_official", search_official_docs_node)
    workflow.add_node("search_community", search_community_node)
    workflow.add_node("grade_docs", grade_documents)
    workflow.add_node("generate", generate)
    workflow.add_node("grade_hallucination", grade_hallucination)
    workflow.add_node("grade_answer", grade_answer_quality)
    
    workflow.add_edge(START, "route")

    workflow.add_conditional_edges(
        "route",
        decide_search_route,
        {
            "official_docs": "search_official",
            "community": "search_community"
        }
    )
    

    workflow.add_edge("search_official", "grade_docs")
    workflow.add_edge("search_community", "grade_docs")
    
    workflow.add_edge("grade_docs", "generate")
    workflow.add_edge("generate", "grade_hallucination")
    workflow.add_edge("grade_hallucination", "grade_answer")
    workflow.add_edge("grade_answer", END)
    
    return workflow.compile()


app = build_graph()

if __name__ == "__main__":
    result = app.invoke({"question": "How to use pandas DataFrame.groupby()?"})
    print(result)
