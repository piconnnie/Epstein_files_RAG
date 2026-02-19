import os
import google.generativeai as genai
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

# Configuration
PERSIST_DIRECTORY = "chroma_db"
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def get_rag_chain_internal():
    print("DEBUG: Initializing RAG Chain with model: gemini-flash-latest")
    # 1. Embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # 2. Vector Store
    if not os.path.exists(PERSIST_DIRECTORY):
        raise FileNotFoundError(f"Vector store not found at {PERSIST_DIRECTORY}. Please run ingest.py first.")
    
    vectorstore = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embeddings)
    
    # 3. Retriever
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # 4. LLM
    llm_provider = os.getenv("LLM_PROVIDER", "google").lower()
    
    if llm_provider == "openai_compatible":
        # Supports xAI (Grok), Groq, DeepSeek, etc.
        from langchain_openai import ChatOpenAI
        
        api_key = os.getenv("OPENAI_API_KEY") # This will store the Grok/Groq key
        base_url = os.getenv("OPENAI_BASE_URL")
        model_name = os.getenv("OPENAI_MODEL_NAME", "grok-beta")
        
        print(f"DEBUG: Initializing RAG Chain with OpenAI-compatible provider: {base_url} model: {model_name}")
        
        llm = ChatOpenAI(
            api_key=api_key,
            base_url=base_url,
            model=model_name,
            temperature=0
        )
    else:
        # Default to Google
        print("DEBUG: Initializing RAG Chain with model: gemini-flash-latest")
        if not GOOGLE_API_KEY:
             raise ValueError("GOOGLE_API_KEY not found. Please set it in .env or use the sidebar.")
        llm = ChatGoogleGenerativeAI(model="gemini-flash-latest", google_api_key=GOOGLE_API_KEY, temperature=0, convert_system_message_to_human=True)

    # 5. Prompt
    prompt_template = """You are an expert AI researcher assisting with the Epstein Files. 
    Your goal is to provide **detailed, comprehensive, and fact-based answers** based ONLY on the provided context.
    
    Guidelines:
    - **Be Thorough**: Do not be brief. Explain the context, who the people are, and what the documents say about them.
    - **No Speculation**: Stick strictly to the text provided.
    - **Citations**: Mention specific details from the text.
    - **Tone**: Professional, objective, and investigative.
    
    Context:
    {context}
    Question:
    {question}

    Answer:
    """
    
    prompt = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    # 6. Chain (LCEL)
    # We want to return both the answer and the source documents.
    # Structure:
    # 1. Retrieve docs based on question
    # 2. Pass context and question to LLM chain
    
    rag_chain_from_docs = (
        RunnablePassthrough.assign(context=(lambda x: format_docs(x["context"])))
        | prompt
        | llm
        | StrOutputParser()
    )

    rag_chain_with_sources = RunnableParallel(
        {"context": retriever, "question": RunnablePassthrough()}
    ).assign(answer=rag_chain_from_docs)

    return rag_chain_with_sources

import time

class ChainAdapter:
    def __init__(self, chain):
        self.chain = chain
        
    def invoke(self, query):
        # Simple retry logic for 429s
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # The underlying LCEL chain accepts a string input directly
                return self.chain.invoke(query)
            except Exception as e:
                error_str = str(e)
                if "429" in error_str and attempt < max_retries - 1:
                    print(f"DEBUG: 429 Error. Retrying in {(attempt + 1) * 5}s...")
                    time.sleep((attempt + 1) * 5)
                    continue
                raise e

def get_rag_chain():
    chain = get_rag_chain_internal()
    return ChainAdapter(chain)
