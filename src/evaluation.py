from sentence_transformers import SentenceTransformer, util
import torch

class RAGEvaluator:
    def __init__(self):
        """
        Initializes the evaluator with a lightweight embedding model.
        """
        # Use the same model as vector store for consistency and efficiency
        self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    def calculate_metrics(self, query: str, answer: str, context: str) -> dict:
        """
        Calculates RAG evaluation metrics.
        """
        metrics = {}
        
        # Encode texts
        query_emb = self.model.encode(query, convert_to_tensor=True)
        answer_emb = self.model.encode(answer, convert_to_tensor=True)
        context_emb = self.model.encode(context, convert_to_tensor=True)
        
        # 1. Answer Relevance (Query vs Answer)
        # How well does the answer address the user's question?
        relevance_score = util.cos_sim(query_emb, answer_emb).item()
        metrics['relevance_score'] = round(relevance_score, 4)
        
        # 2. Faithfulness / Context Adherence (Answer vs Context)
        # Is the answer grounded in the retrieved context?
        faithfulness_score = util.cos_sim(answer_emb, context_emb).item()
        metrics['faithfulness_score'] = round(faithfulness_score, 4)
        
        return metrics

if __name__ == "__main__":
    evaluator = RAGEvaluator()
    m = evaluator.calculate_metrics("What is X?", "X is Y.", "X is Y because Z.")
    print(m)
