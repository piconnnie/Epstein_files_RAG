import streamlit as st
import os
from rag_engine import get_rag_chain

st.set_page_config(page_title="Epstein Files RAG", page_icon="⚖️", layout="wide")

st.title("⚖️ Epstein Files RAG MVP")
st.markdown("""
**Public AI Assistant** for exploring the Epstein Files. 
*Ask questions grounded in the `teyler/epstein-files-20k` dataset.*
""")

# Sidebar for API Key
# Simple toggle for provider
provider = st.sidebar.selectbox("Select LLM Provider", ["Google Gemini", "xAI (Grok)", "Groq"])

if provider == "Google Gemini":
    if "GOOGLE_API_KEY" not in os.environ:
        api_key = st.sidebar.text_input("Enter Google API Key", type="password")
        if api_key:
            os.environ["GOOGLE_API_KEY"] = api_key
    os.environ["LLM_PROVIDER"] = "google"

elif provider == "xAI (Grok)":
    if "OPENAI_API_KEY" not in os.environ:
        api_key = st.sidebar.text_input("Enter xAI API Key", type="password")
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
    
    # Model selection for Grok
    grok_models = ["grok-beta", "grok-vision-beta"]
    selected_model = st.sidebar.selectbox("Select Model", grok_models + ["Other (Manual Input)"])
    
    if selected_model == "Other (Manual Input)":
        model_name = st.sidebar.text_input("Enter Model Name", value="grok-beta")
    else:
        model_name = selected_model
        
    os.environ["OPENAI_MODEL_NAME"] = model_name
    os.environ["OPENAI_BASE_URL"] = "https://api.x.ai/v1"
    os.environ["LLM_PROVIDER"] = "openai_compatible"

elif provider == "Groq":
    if "OPENAI_API_KEY" not in os.environ:
        api_key = st.sidebar.text_input("Enter Groq API Key", type="password")
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key

    # Model selection for Groq
    groq_models = [
        "llama-3.3-70b-versatile",
        "llama-3.1-8b-instant",
        "llama-3.1-70b-versatile", # Kept for backward compat if needed, though docs emphasize 3.3
        "mixtral-8x7b-32768", # Popular fallback
        "gemma2-9b-it",
    ]
    selected_model = st.sidebar.selectbox("Select Model", groq_models + ["Other (Manual Input)"])
    
    if selected_model == "Other (Manual Input)":
        model_name = st.sidebar.text_input("Enter Model Name", value="llama-3.3-70b-versatile")
    else:
        model_name = selected_model
        
    os.environ["OPENAI_MODEL_NAME"] = model_name
    os.environ["OPENAI_BASE_URL"] = "https://api.groq.com/openai/v1"
    os.environ["LLM_PROVIDER"] = "openai_compatible"

if "GOOGLE_API_KEY" not in os.environ and "OPENAI_API_KEY" not in os.environ:
    st.warning(f"Please provide an API Key for {provider} to proceed.")
    st.stop()

# Initialize Chain
# Check if vector store exists, if not, ingest data (for Cloud Deployment)
PERSIST_DIRECTORY = "chroma_db"

@st.cache_resource
def ensure_vector_store():
    if not os.path.exists(PERSIST_DIRECTORY):
        with st.spinner("Downloading and processing data (First run only)..."):
            # Import here to avoid circular imports if any, or just for cleanliness
            from ingest import ingest_data
            ingest_data()
    return True

# Ensure data is ready
ensure_vector_store()

@st.cache_resource
def load_chain_v3():
    # Force reload of chain to pick up any new configuration
    return get_rag_chain()

try:
    qa_chain = load_chain_v3()
except ValueError as e:
    st.error(str(e))
    st.stop()
except Exception as e:
    st.error(f"Failed to load RAG engine: {e}")
    st.stop()

# Session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("Ask about the Epstein Files..."):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        try:
            # Prepare chat history for the chain (if supported later, for now just query)
            # We can pass history if we update rag_engine.py, but for now let's just do single turn
            # or simple concatenation if needed.
            
            # Run RAG
            # Initialize chain if not already done (it is cached)
            chain = load_chain_v3()
            
            with st.spinner("Searching and generating answer..."):
                response_obj = chain.invoke(prompt)
                
                # Handle different return types from LCEL
                if isinstance(response_obj, dict):
                    answer = response_obj.get('answer', '')
                    sources = response_obj.get('context', [])
                else:
                    answer = str(response_obj)
                    sources = []

                # Stream response (simulated for now, or just print)
                message_placeholder.markdown(answer)
                full_response = answer
                
                # Display sources in an expander
                if sources:
                    with st.expander("📚 Sources & Evidence"):
                        for i, doc in enumerate(sources):
                            source_name = doc.metadata.get("source", "Unknown")
                            st.markdown(f"**Source {i+1}:** `{source_name}`")
                            st.caption(doc.page_content[:300] + "...")
                             # Attempt to make a clickable link if it's a known format
                            if source_name.startswith("http"):
                                st.markdown(f"[Open Source]({source_name})")
                            st.divider()

        except Exception as e:
            full_response = f"Error generating answer: {e}"
            message_placeholder.error(full_response)
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": full_response})

st.markdown("---")
st.caption("Disclaimer: This tool answers questions based only on publicly available documents. It does not make claims, judgments, or allegations.")
