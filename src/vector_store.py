import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from typing import List, Dict, Any
from langchain_core.documents import Document

# Configuration
CHROMA_DATA_PATH = "data/chroma_db"
COLLECTION_NAME = "clinical_notes"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

def get_vectorstore():
    """
    Returns the LangChain Chroma vector store.
    """
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    
    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_DATA_PATH
    )
    return vectorstore

def add_documents_to_db(documents: List[Dict[str, Any]]):
    """
    Adds documents to the LangChain Chroma vector store.
    """
    print(f"Initializing Vector Store with {len(documents)} documents...")
    
    # Convert dicts to LangChain Documents
    langchain_docs = []
    for doc in documents:
        langchain_docs.append(Document(
            page_content=doc['page_content'],
            metadata=doc['metadata']
        ))
    
    # Initialize and add
    vectorstore = get_vectorstore()
    
    # Add in batches (LangChain handles this, but good to be explicit if needed)
    vectorstore.add_documents(langchain_docs)
    print("Successfully added documents to ChromaDB via LangChain.")

if __name__ == "__main__":
    # Test and Populate DB
    try:
        # Import ingestion logic here to avoid circular imports if any
        # (Though currently there are none, it's safe practice for scripts)
        import sys
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from src.ingestion import load_clinical_notes

        # Define path to data
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        
        # Try finding data in the standard location or fallback
        data_path = os.path.join(project_root, "data", "raw")
        if not os.path.exists(data_path) or not os.listdir(data_path):
             # Fallback logic
             workspace_root = os.path.dirname(project_root)
             potential_path = os.path.join(workspace_root, "mimic-iv-ext-direct-1.0.0", "Finished")
             if os.path.exists(potential_path):
                 print(f"Data not found in {data_path}, using {potential_path}")
                 data_path = potential_path

        # 1. Load Documents
        docs = load_clinical_notes(data_path)
        
        if docs:
            # 2. Add to Vector Store
            add_documents_to_db(docs)
            
            # 3. Verify
            vs = get_vectorstore()
            print(f"Vector Store initialized successfully. Collection: {vs._collection.name}")
            
    except Exception as e:
        print(f"Error: {e}")
