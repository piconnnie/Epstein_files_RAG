import os
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

PERSIST_DIRECTORY = "chroma_db"

def test_vector_store():
    if not os.path.exists(PERSIST_DIRECTORY):
        print("FAIL: Vector store directory not found.")
        return

    print(f"Loading vector store from {PERSIST_DIRECTORY}...")
    try:
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embeddings)
        print("Vector store loaded successfully.")
        
        # Test retrieval
        print("Testing retrieval...")
        retriever = vectorstore.as_retriever(search_kwargs={"k": 1})
        docs = retriever.get_relevant_documents("Epstein")
        
        if docs:
            print("SUCCESS: Retrieved documents.")
            print(f"First doc content snippet: {docs[0].page_content[:100]}...")
        else:
            print("WARNING: No documents retrieved (might be empty query or DB).")
            
    except Exception as e:
        print(f"FAIL: Error loading/querying vector store: {e}")

if __name__ == "__main__":
    test_vector_store()
