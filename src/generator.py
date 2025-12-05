import os
from langchain_huggingface import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

class RAGChain:
    def __init__(self):
        """
        Initializes the RAG chain with a local Flan-T5 model.
        """
        print("Initializing Local LLM (google/flan-t5-base)...")
        model_id = "google/flan-t5-base"
        
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
        
        pipe = pipeline(
            "text2text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=512,
            temperature=0.5,
            do_sample=True
        )
        
        self.llm = HuggingFacePipeline(pipeline=pipe)

        # Define Prompt (Concise to save tokens)
        self.prompt = ChatPromptTemplate.from_template("""
Answer based on context.

Context:
{context}

Question: {question}

Answer:
""")
        
        self.output_parser = StrOutputParser()

    def generate_answer(self, query: str, context_str: str) -> str:
        """
        Generates answer using the chain.
        """
        # Truncate context to fit within 512 tokens (approx 2000 chars)
        # Reserve ~100 chars for query and template
        max_context_chars = 1800
        if len(context_str) > max_context_chars:
            context_str = context_str[:max_context_chars] + "... (truncated)"
        
        # Create a simple chain: prompt -> llm -> output_parser
        chain = self.prompt | self.llm | self.output_parser
        
        try:
            response = chain.invoke({
                "context": context_str,
                "question": query
            })
            return response
        except Exception as e:
            return f"Error generating answer: {e}"

if __name__ == "__main__":
    rag = RAGChain()
    print("Local RAG Chain initialized.")
