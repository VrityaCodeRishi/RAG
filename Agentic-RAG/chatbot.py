# UI quickly generated from Cursor. This is integrated with my RAG system.

import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage
from rag import graph

load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Anubhav's Hobby Alter EGO",
    page_icon="🤖",
    layout="wide"
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Title and description
st.title("ANUBHAV'S HOBBY ALTER EGO")
st.markdown("""
Ask me about anime or games! I can share my personal experiences and thoughts about them.
""")

# Sidebar with information
with st.sidebar:
    st.header("ℹ️ About")
    st.markdown("""
    This chatbot uses a RAG (Retrieval-Augmented Generation) system to answer questions about:
    - **Anime experiences** - Personal thoughts and ratings
    - **Game experiences** - Gaming experiences and reviews
    
    The system uses:
    - Vector databases for semantic search
    - Web search for entity identification
    - LLM for generating responses
    """)
    
    st.header("💡 Example Questions")
    st.markdown("""
    - "Did you like 'God of War'?"
    - "What do you think about 'Attack on Titan'?"
    - "Tell me about your experience with 'The Last of Us'"
    - "What's your favorite anime character?"
    """)
    
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.session_state.chat_history = []
        st.rerun()

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask me about anime or games..."):
    # Add user message to chat
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate response using the RAG graph
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                # Invoke the graph with the user's query
                result = graph.invoke({"messages": [HumanMessage(content=prompt)]})
                
                # Extract the final response
                response = result["messages"][-1].content
                
                # Display the response
                st.markdown(response)
                
                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": response})
                
            except Exception as e:
                error_message = f"Sorry, I encountered an error: {str(e)}"
                st.error(error_message)
                st.session_state.messages.append({"role": "assistant", "content": error_message})
