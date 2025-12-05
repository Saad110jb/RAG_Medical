import os
import json
from typing import List, Dict, Any

def extract_text_and_metadata(data: Dict[str, Any]) -> tuple[str, List[str]]:

    """
    Recursively extracts text from 'input' keys and collects metadata keys.
    """
    text_content = []
    diagnoses = []

    def _recurse(node):
        if isinstance(node, dict):
            for key, value in node.items():
                # Check if key is an input key (e.g., input1, input2, Input6)
                if key.lower().startswith("input"):
                    if isinstance(value, str):
                        text_content.append(value)
                    elif isinstance(value, dict):
                         _recurse(value)
                else:
                    # It's a diagnosis/reasoning key
                    # We store these as metadata
                    diagnoses.append(key)
                    _recurse(value)
        elif isinstance(node, list):
            for item in node:
                _recurse(item)

    _recurse(data)
    _recurse(data)
    
    # Join text content
    raw_text = "\n".join(text_content)
    
    # Apply preprocessing
    from src.preprocessing import clean_text
    cleaned_text = clean_text(raw_text)
    
    return cleaned_text, diagnoses

def load_clinical_notes(base_path: str) -> List[Dict[str, Any]]:
    """
    Walks through the dataset and parses clinical notes.
    """
    documents = []
    
    if not os.path.exists(base_path):
        print(f"Warning: Data path {base_path} does not exist.")
        return []

    print(f"Scanning directory: {base_path}")
    
    # Walk through the directory
    for root, dirs, files in os.walk(base_path):
        for file in files:
            if not file.endswith(".json"):
                continue
            
            file_path = os.path.join(root, file)
            
            # Determine Condition and SubDiagnosis based on path
            # Expected structure: base_path/Condition/SubDiagnosis/file.json
            # or base_path/Condition/file.json
            
            rel_path = os.path.relpath(root, base_path)
            parts = rel_path.split(os.sep)
            
            # Handle cases where we might be at the root or deeper
            if rel_path == ".":
                condition = "Unknown"
                sub_diagnosis = "None"
            else:
                condition = parts[0]
                sub_diagnosis = parts[1] if len(parts) > 1 else "None"
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                text, detailed_diagnoses = extract_text_and_metadata(data)
                
                # Metadata construction
                metadata = {
                    "condition": condition,
                    "sub_diagnosis": sub_diagnosis,
                    "source": file,
                    # Store detailed diagnoses as a string or list? 
                    # ChromaDB metadata values must be str, int, float, or bool. Lists are not supported directly in standard Chroma.
                    # We will join them with a separator.
                    "reasoning_chain": " || ".join(detailed_diagnoses)
                }
                
                documents.append({
                    "page_content": text,
                    "metadata": metadata
                })
                
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                
    return documents

if __name__ == "__main__":
    # Test run
    # Define paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    data_path = os.path.join(project_root, "data", "raw")
    
    # Check if data/raw has content, if not, try to find the original dataset
    if not os.path.exists(data_path) or not os.listdir(data_path):
        
        # Fallback to the mimic folder in the workspace root if available
        # Assuming workspace root is two levels up from src (Clinical-RAG-Direct/src -> Clinical-RAG-Direct -> Workspace)
        # Wait, project_root is Clinical-RAG-Direct. Workspace is one level up.
        workspace_root = os.path.dirname(project_root)
        potential_path = os.path.join(workspace_root, "mimic-iv-ext-direct-1.0.0", "Finished")
        if os.path.exists(potential_path):
            print(f"Data not found in {data_path}, using {potential_path}")
            data_path = potential_path
    
    docs = load_clinical_notes(data_path)
    print(f"Successfully loaded {len(docs)} documents.")
    
    if docs:
        print("\n--- Sample Document ---")
        print(f"Source: {docs[0]['metadata']['source']}")
        print(f"Condition: {docs[0]['metadata']['condition']}")
        print(f"Sub-Diagnosis: {docs[0]['metadata']['sub_diagnosis']}")
        print(f"Content Preview: {docs[0]['page_content'][:200]}...")
