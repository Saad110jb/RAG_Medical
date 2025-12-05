import re

def clean_text(text: str) -> str:
    """
    Cleans clinical text by:
    1. Removing excessive whitespace/newlines.
    2. Normalizing special characters.
    3. Removing generic headers if needed.
    """
    if not text:
        return ""
    
    # Replace multiple newlines/tabs with a single space
    text = re.sub(r'\s+', ' ', text)
    
    # Remove leading/trailing whitespace
    text = text.strip()
    
    # Optional: Remove specific artifacts if known (e.g., "Input1:", "Input2:")
    # For now, general whitespace cleaning is the most important for RAG
    
    return text
