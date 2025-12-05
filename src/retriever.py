from src.vector_store import get_vectorstore
from typing import List, Dict, Any

class LangChainRetriever:
    def __init__(self):
        """
        Initializes the LangChain retriever.
        """
        self.vectorstore = get_vectorstore()

    def get_relevant_documents(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
        """
        Retrieves documents and returns them with metadata and scores.
        """
        # Perform similarity search with score
        # Note: Chroma returns distance (lower is better), but LangChain unifies this usually.
        # We'll use similarity_search_with_score
        docs_and_scores = self.vectorstore.similarity_search_with_score(query, k=k)
        
        results = []
        for doc, score in docs_and_scores:
            results.append({
                "content": doc.page_content,
                "metadata": doc.metadata,
                "score": score
            })
        return results

    def format_docs(self, docs: List[Dict[str, Any]]) -> str:
        """
        Formats documents for the LLM context.
        """
        context = ""
        for i, d in enumerate(docs):
            context += f"--- Document {i+1} ---\n"
            context += f"Condition: {d['metadata'].get('condition')}\n"
            context += f"Content: {d['content']}\n\n"
        return context

if __name__ == "__main__":
    retriever = LangChainRetriever()
    results = retriever.get_relevant_documents("chest pain")
    print(f"Retrieved {len(results)} docs.")
    print(results[0]['content'][:100])
