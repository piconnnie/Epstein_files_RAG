import os
from datasets import load_dataset
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

# Configuration
DATASET_NAME = "teyler/epstein-files-20k"
PERSIST_DIRECTORY = "chroma_db"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

def ingest_data():
    print(f"Loading dataset: {DATASET_NAME}...")
    try:
        dataset = load_dataset(DATASET_NAME, split="train")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    print(f"Dataset loaded. {len(dataset)} records found.")
    
    # Simple check for text column
    text_column = "text"
    if "text" not in dataset.column_names:
        # Fallback if 'text' isn't the name, try to find a string column
        for col in dataset.column_names:
            if isinstance(dataset[0][col], str):
                text_column = col
                break
    
    print(f"Using column '{text_column}' for text content.")

    print("Initializing embedding model...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    print("Splitting text...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        add_start_index=True,
    )

    print(f"Created text splitter.")
    
    # Create Vector Store (Initialize empty)
    print(f"Creating Chroma vector store in {PERSIST_DIRECTORY}...")
    if os.path.exists(PERSIST_DIRECTORY):
        print("Removing existing vector store...")
        import shutil
        shutil.rmtree(PERSIST_DIRECTORY)

    # Initialize VectorStore with the first batch or empty? 
    # We can't init empty easily with from_documents, so we init with the class constructor
    # But for simplicity, let's collect the first batch, create the DB, then add the rest.
    
    vectorstore = None
    BATCH_SIZE = 500
    batch_docs = []
    total_chunks = 0
    MAX_DOCS = None
    
    import re
    # More permissive regex for filenames with spaces or other chars
    filename_pattern = r'^([^\,]+\.txt),(.*)$'

    print("Starting batched ingestion...")

    for i, record in enumerate(dataset):
        if MAX_DOCS and i >= MAX_DOCS:
            break
        
        text = record.get(text_column, "")
        if not text or not isinstance(text, str):
            continue
            
        # Parse filename if present
        match = re.match(filename_pattern, text, re.DOTALL)
        if match:
            source = match.group(1)
            content = match.group(2)
            if content.startswith('"') and content.endswith('"'):
                content = content[1:-1]
        else:
            source = "unknown"
            content = text
            
        # Split content into chunks
        chunks = text_splitter.split_text(content)
        for chunk in chunks:
            batch_docs.append(Document(page_content=chunk, metadata={"source": source}))
            
        # Process batch
        if len(batch_docs) >= BATCH_SIZE:
            if vectorstore is None:
                vectorstore = Chroma.from_documents(
                    documents=batch_docs, 
                    embedding=embeddings, 
                    persist_directory=PERSIST_DIRECTORY
                )
            else:
                vectorstore.add_documents(batch_docs)
            
            total_chunks += len(batch_docs)
            print(f"Processed {total_chunks} chunks...", flush=True)
            batch_docs = [] # Clear memory

    # Process remaining documents
    if batch_docs:
        if vectorstore is None:
            vectorstore = Chroma.from_documents(
                documents=batch_docs, 
                embedding=embeddings, 
                persist_directory=PERSIST_DIRECTORY
            )
        else:
            vectorstore.add_documents(batch_docs)
        total_chunks += len(batch_docs)
        print(f"Processed final batch. Total chunks: {total_chunks}")
    
    print(f"Ingestion complete. Vector store saved to '{PERSIST_DIRECTORY}'")

if __name__ == "__main__":
    ingest_data()
