from dotenv import load_dotenv
from openai import OpenAI
from mem0 import MemoryClient
import streamlit as st
from agent_router import router
from agents import AgentFactory
from supports import retrieve

load_dotenv("../.env")
client = OpenAI()
memory_client = MemoryClient()

def chat_with_ai(user_prompt: str, user_id="default_user") -> str:

    # Step 1: Classify the request
    with st.spinner("🔍 Analyzing your request..."):
        classification = router.classify_request(user_prompt)

    # Step 2: Extract specific parameters
    extracted_params = router.extract_specific_params(classification, user_prompt)

    # Step 3: add relevant memories for context
    relevant_memories = []
    with st.spinner("🧠 Retrieving relevant memories..."):
        relevant_memories = memory_client.search(query=user_prompt, user_id=user_id, limit=3)

    if relevant_memories:
        memory_context = "\n".join(f"- {entry['memory']}" for entry in relevant_memories)
        extracted_params["memory_context"] = memory_context

    # Step 4: Perform RAG search to get relevant documents
    retrieve_results = []
    with st.spinner("🔍 Retrieving relevant documents..."):
        retrieve_results = retrieve(user_prompt, top_k=3)
    if retrieve_results:
        extracted_params["rag_context"] = "\n".join(f"- {doc}" for doc in retrieve_results)

    # Step 5: Show routing information to user
    with st.expander("🤖 Agent Routing", expanded=False):
        st.write(f"**Selected Agent:** {classification.agent_type.value}")
        st.write(f"**Confidence:** {classification.confidence:.2f}")
        st.write(f"**Reasoning:** {classification.reasoning}")

    # Step 6: Create and use the appropriate agent
    agent = AgentFactory.create_agent(classification.agent_type.value)

    # Step 7: Process the request with the specialized agent
    response = agent.process_request(user_prompt, extracted_params)

    memory_messages = [
        {"role": "user", "content": user_prompt},
        {"role": "assistant", "content": response},
    ]

    memory_client.add(
        memory_messages,
        user_id=user_id,
        metadata={"app_id": "customer-support"},
        output_format="v1.1"
    )

    return response

# ========== Main Chat Loop ==========
def main():
    st.set_page_config(
        page_title="Multi-Agent Assistant",
        page_icon="🤖",
        layout="wide"
    )
    
    st.title("🤖 Multi-Agent Assistant")
    st.markdown("**Intelligent multi-agent system with specialized developer and customer support capabilities**")
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat messages from history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("What can I help you with today? (e.g., 'analyze code in project/src', 'check my order #12345', 'explain API best practices')"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate AI response
        with st.chat_message("assistant"):
            response = chat_with_ai(prompt)
            st.session_state.messages.append({"role": "assistant", "content": response})
    
    # Sidebar with instructions and memory management
    with st.sidebar:
        st.header("🤖 Multi-Agent Assistant")
        st.markdown("""
        This system provides **intelligent multi-agent assistance** with specialized capabilities:
        
        **🛠️ Developer Tasks**
        - `analyze code in project/src`
        - `check for bugs in main.py`
        - `review code quality in directory`
        - `analyze logs in project/logs`
        - `find errors in error.log`
        - `check performance issues in logs`

        **🎧 Customer Support**
        - `check my order status for order #12345`
        - `what is your refund policy?`
        - `help me with my account issues`
        - `track my recent purchases`
        - `cancel my subscription`

        **Fix Proposals**
        - `analyze and fix bugs in project`
        - `propose improvements for code`
        - `suggest fixes for log errors`

        **🎯 General Assistance**
        - Get help with various development and support tasks
        - Ask questions about best practices
        - Request clarification on any topic
        """)
        
        st.header("Example Requests")
        st.code("""
# Developer Tasks
analyze code in project/src directory
check for bugs in index.js
review code quality in main.py
analyze logs in project/logs
find errors in error.log file

# Customer Support
check my order status for order #12345
what is your refund policy?
help me with my account issues
track my recent purchases
cancel my subscription

# General Questions
what are the best practices for API design?
how can I improve my code quality?
explain the difference between async and sync
        """)
        
        st.info("💡 **Smart Routing**: The system automatically routes your request to the most appropriate specialist agent!")
        
        if st.button("🗑️ Clear Chat History"):
            st.session_state.messages = []
            st.rerun()

if __name__ == "__main__":
    main()
