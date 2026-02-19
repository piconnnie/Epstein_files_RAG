
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
import os

PERSIST_DIRECTORY = "chroma_db"

def inspect():
    if not os.path.exists(PERSIST_DIRECTORY):
        print("ChromaDB not found locally.")
        return

    print("Loading ChromaDB...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embeddings)
    
    print("Searching for 'Modi' in vector store (k=10)...")
    results = vectorstore.similarity_search("Modi", k=10)
    
    print(f"Found {len(results)} results.")
    for i, doc in enumerate(results):
        print(f"\n--- Result {i+1} ---")
        print(f"Source Metadata: {doc.metadata}")
        print(f"Content Preview: {doc.page_content[:200]}...")
        if "Modi" in doc.page_content:
            print(">>> 'Modi' FOUND in content!")
        else:
            print(">>> 'Modi' NOT found in content (semantic match only?)")

    print("\n\nChecking a random document for metadata structure...")
    # Hack to get a random doc? 
    # langchain chroma doesn't expose 'get all', but we can search for a common word like "the"
    common = vectorstore.similarity_search("Epstein", k=1)
    if common:
        print(f"Sample Metadata: {common[0].metadata}")

if __name__ == "__main__":
    inspect()
