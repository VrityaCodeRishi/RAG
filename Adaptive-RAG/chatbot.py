# Streamlit UI for Adaptive RAG Technical Documentation Assistant
# This imports the graph builder from rag.py
# I generated this UI code from Cursor so the streamlit code is purely vibe coded.

import streamlit as st
from dotenv import load_dotenv
from rag import app  # Import the compiled graph from rag.py

load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Technical Documentation Assistant - Adaptive RAG",
    page_icon="💻",
    layout="wide"
)

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Title and description
st.title("💻 Technical Documentation Assistant")
st.markdown("""
Ask me about programming, APIs, frameworks, or troubleshooting!
The system uses **Adaptive RAG** with self-evaluation to provide accurate answers from official docs and community solutions.
""")

# Sidebar with information
with st.sidebar:
    st.header("ℹ️ About Adaptive RAG")
    st.markdown("""
    This chatbot uses **Adaptive RAG** (Retrieval-Augmented Generation) with multiple self-evaluation steps:
    
    **1. Route Evaluation** 🔀
    - Decides: Official docs or community solutions?
    
    **2. Document Relevance** 📊
    - Filters irrelevant search results
    
    **3. Answer Generation** ✍️
    - Creates answer from filtered results
    
    **4. Hallucination Check** ✅
    - Verifies answer is grounded in facts
    
    **5. Answer Quality** 🎯
    - Ensures answer addresses the question
    
    The LLM evaluates itself at each step!
    """)
    
    st.header("💡 Example Questions")
    st.markdown("""
    **Official Documentation:**
    - "How to use pandas DataFrame.groupby()?"
    - "React useState hook documentation"
    - "Python list comprehension syntax"
    
    **Community Solutions:**
    - "How to fix ModuleNotFoundError?"
    - "Best way to handle async errors in Python"
    - "Stack Overflow solution for TypeError"
    """)
    
    st.header("🔍 Search Sources")
    st.markdown("""
    - **Official Docs**: Python, pandas, React documentation
    - **Community**: Stack Overflow, GitHub, blogs
    """)
    
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask me about programming, APIs, or troubleshooting..."):
    # Add user message to chat
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate response using the Adaptive RAG graph
    with st.chat_message("assistant"):
        with st.spinner("🔍 Searching and evaluating..."):
            try:
                # Invoke the graph with the user's question
                # The graph expects {"question": "..."} format
                result = app.invoke({"question": prompt})
                
                # Extract the final response from generation field
                response = result["generation"]
                
                # Display the response
                st.markdown(response)
                
                # Optional: Show routing decision (if you want to display it)
                if "route_decision" in result:
                    route = result["route_decision"]
                    if route == "official_docs":
                        st.caption("📚 Used: Official Documentation")
                    else:
                        st.caption("💬 Used: Community Solutions")
                
                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": response})
                
            except Exception as e:
                error_message = f"Sorry, I encountered an error: {str(e)}"
                st.error(error_message)
                st.session_state.messages.append({"role": "assistant", "content": error_message})

