# UI for Corrective RAG Recipe Assistant I generated from Cursor. I asked it to import graph builder we created directly.

import streamlit as st
from dotenv import load_dotenv
from rag import builder

load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Recipe Assistant - Corrective RAG",
    page_icon="🍳",
    layout="wide"
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Title and description
st.title("🍳 Recipe Assistant - Corrective RAG")
st.markdown("""
Ask me about recipes, cooking techniques, ingredient substitutions, or dietary modifications!
The system uses Corrective RAG to provide accurate answers from our knowledge base and real-time web search.
""")

# Sidebar with information
with st.sidebar:
    st.header("ℹ️ About")
    st.markdown("""
    This chatbot uses **Corrective RAG** (Retrieval-Augmented Generation) to answer questions about:
    - **Recipes** - Find recipes for any dish
    - **Cooking Techniques** - Learn cooking methods
    - **Ingredient Substitutions** - Find alternatives for ingredients
    - **Dietary Modifications** - Adapt recipes for dietary needs
    
    The system:
    - ✅ Retrieves from vector database
    - ✅ Grades document relevance
    - ✅ Searches web when needed
    - ✅ Combines knowledge for accurate answers
    """)
    
    st.header("💡 Example Questions")
    st.markdown("""
    - "How to make chocolate cake?"
    - "What can I substitute for eggs in baking?"
    - "Vegan pasta recipe"
    - "How to cook perfect steak?"
    - "Trending air fryer recipes 2024"
    - "Gluten-free bread recipe"
    """)
    
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask me about recipes, cooking, or ingredients..."):
    # Add user message to chat
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate response using the Corrective RAG builder
    with st.chat_message("assistant"):
        with st.spinner("Cooking up an answer..."):
            try:
                # Invoke the builder with the user's question
                # Corrective RAG uses {"question": "..."} format
                result = builder.invoke({"question": prompt})
                
                # Extract the final response from generation field
                response = result["generation"]
                
                # Display the response
                st.markdown(response)
                
                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": response})
                
            except Exception as e:
                error_message = f"Sorry, I encountered an error: {str(e)}"
                st.error(error_message)
                st.session_state.messages.append({"role": "assistant", "content": error_message})

