import os
import time
from typing import List, Dict, Any, Generator

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

# Singleton-like cache for embeddings
_embeddings_cache = None

def get_embeddings():
    global _embeddings_cache
    if _embeddings_cache is None:
        print("DEBUG: Initializing Embedding Model (sentence-transformers/all-MiniLM-L6-v2)")
        _embeddings_cache = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return _embeddings_cache

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

class ChainAdapter:
    def __init__(self, chain, retriever, rag_chain_from_docs):
        self.chain = chain
        self.retriever = retriever
        self.rag_chain_from_docs = rag_chain_from_docs
        
    def stream(self, query: str) -> Generator[Dict[str, Any], None, None]:
        """
        Streams the response to allow app.py to display sources immediately
        then stream the answer.
        """
        try:
            # 1. Retrieval
            print(f"DEBUG: Retrieving documents for query: {query}")
            docs = self.retriever.invoke(query)
            yield {"type": "context", "content": docs}
            
            # 2. Generation
            print(f"DEBUG: Starting LLM stream...")
            
            # Since we already have the docs, we can manually feed them into the rag_chain_from_docs
            # rag_chain_from_docs expects a dict with "context" and "question"
            # Since it already has a lambda to format docs, we pass the raw docs list
            input_dict = {"context": docs, "question": query}
            
            for chunk in self.rag_chain_from_docs.stream(input_dict):
                yield {"type": "answer_chunk", "content": chunk}
                    
        except Exception as e:
            import traceback
            traceback.print_exc()
            yield {"type": "error", "content": str(e)}

    def invoke(self, query):
        return self.chain.invoke(query)

def get_rag_chain():
    # 1. Embeddings
    embeddings = get_embeddings()

    # 2. Vector Store
    if not os.path.exists(PERSIST_DIRECTORY):
        raise FileNotFoundError(f"Vector store not found at {PERSIST_DIRECTORY}. Please run ingest.py first.")
    
    vectorstore = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embeddings)
    
    # 3. Retriever
    retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

    chain, rag_chain_from_docs = get_rag_chain_internal(retriever)
    return ChainAdapter(chain, retriever, rag_chain_from_docs)

def get_rag_chain_internal(retriever):
    # 4. LLM
    llm_provider = os.getenv("LLM_PROVIDER", "google").lower()
    
    if llm_provider == "openai_compatible":
        from langchain_openai import ChatOpenAI
        api_key = os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("OPENAI_BASE_URL")
        model_name = os.getenv("OPENAI_MODEL_NAME", "grok-beta")
        
        print(f"DEBUG: Initializing LLM: {model_name}")
        
        llm = ChatOpenAI(
            api_key=api_key,
            base_url=base_url,
            model=model_name,
            temperature=0,
            streaming=True
        )
    else:
        # Default to Google
        if not GOOGLE_API_KEY:
             raise ValueError("GOOGLE_API_KEY not found. Please set it in .env or use the sidebar.")
        print("DEBUG: Initializing LLM: gemini-flash-latest")
        llm = ChatGoogleGenerativeAI(
            model="gemini-flash-latest", 
            google_api_key=GOOGLE_API_KEY, 
            temperature=0, 
            convert_system_message_to_human=True,
            streaming=True
        )

    # 5. Prompt
    prompt_template = """You are an expert AI researcher assisting with the Epstein Files. 
    Your goal is to provide **detailed, comprehensive, and fact-based answers** based ONLY on the provided context.
    
    Guidelines:
    - **Be Thorough**: Explain the context, the people involved, and what the documents say about them.
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
    rag_chain_from_docs = (
        RunnablePassthrough.assign(context=(lambda x: format_docs(x["context"])))
        | prompt
        | llm
        | StrOutputParser()
    )

    rag_chain_with_sources = RunnableParallel(
        {"context": retriever, "question": RunnablePassthrough()}
    ).assign(answer=rag_chain_from_docs)

    return rag_chain_with_sources, rag_chain_from_docs
