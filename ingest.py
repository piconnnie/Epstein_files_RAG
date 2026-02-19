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

    docs = []
    # Limit for MVP to avoid huge time/memory usage if dataset is massive, 
    # but 20k records is manageable. 
    # Let's process in batches or just all at once if memory allows.
    # For safety/speed in MVP demo, let's take first 1000 or so if it's very slow, 
    # but the requirement is "few million lines", 20k docs * ~10 chunks = ~200k vectors. 
    # Chroma local might be slow with 200k. Let's try 2000 documents first for the MVP.
    
    MAX_DOCS = None # Process all ~20k documents
    if MAX_DOCS:
        print(f"Processing first {MAX_DOCS} documents for MVP speed...")
    else:
        print("Processing entire dataset...")
    
    import re
    # More permissive regex for filenames with spaces or other chars
    filename_pattern = r'^([^\,]+\.txt),(.*)$'

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
            # Remove wrapping quotes if present
            if content.startswith('"') and content.endswith('"'):
                content = content[1:-1]
        else:
            source = "unknown"
            content = text
            
        # Split content into chunks immediately with metadata
        chunks = text_splitter.split_text(content)
        for chunk in chunks:
            docs.append(Document(page_content=chunk, metadata={"source": source}))

    print(f"Created {len(docs)} chunks.")
    
    # Create Vector Store
    print(f"Creating Chroma vector store in {PERSIST_DIRECTORY}...")
    if os.path.exists(PERSIST_DIRECTORY):
        print("Removing existing vector store...")
        import shutil
        shutil.rmtree(PERSIST_DIRECTORY)

    vectorstore = Chroma.from_documents(
        documents=docs, 
        embedding=embeddings, 
        persist_directory=PERSIST_DIRECTORY
    )
    
    print(f"Ingestion complete. Vector store saved to '{PERSIST_DIRECTORY}'")

if __name__ == "__main__":
    ingest_data()
