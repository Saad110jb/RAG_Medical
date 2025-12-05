import streamlit as st
from src.retriever import LangChainRetriever
from src.generator import RAGChain
from src.evaluation import RAGEvaluator

# Page Config
st.set_page_config(page_title="DIRECT: Diagnostic Reasoning (LangChain)", layout="wide")

# Title
st.title("🩺 DIRECT: RAG for Diagnostic Reasoning")
st.markdown("Powered by **LangChain** & **Local LLM (Flan-T5)**")

# Sidebar
with st.sidebar:
    st.header("Configuration")
    k_retrieval = st.slider("Number of Documents to Retrieve", min_value=1, max_value=5, value=2)
    st.info("Using **Google Flan-T5** (Local) for generation. No API Key required.")

# Initialize components
@st.cache_resource
def load_components():
    retriever = LangChainRetriever()
    rag_chain = RAGChain()
    evaluator = RAGEvaluator()
    return retriever, rag_chain, evaluator

try:
    retriever, rag_chain, evaluator = load_components()
    st.success("LangChain Components Loaded Successfully!")
except Exception as e:
    st.error(f"Error loading components: {e}")
    st.stop()

# Main Interface
query = st.text_input("Enter your diagnostic query:", placeholder="e.g., Why does this patient have hypoxia?")

if st.button("Analyze Case"):
    if not query:
        st.warning("Please enter a query.")
    else:
        # 1. Retrieval Step
        st.subheader("1. Retrieval Step")
        with st.spinner("Retrieving documents..."):
            retrieved_docs = retriever.get_relevant_documents(query, k=k_retrieval)
            
            if not retrieved_docs:
                st.error("No relevant documents found.")
            else:
                # Show retrieved docs in an expandable section
                with st.expander(f"📄 Retrieved {len(retrieved_docs)} Documents (Click to View)", expanded=True):
                    for i, doc in enumerate(retrieved_docs):
                        st.markdown(f"**Document {i+1}** (Score: {doc['score']:.4f})")
                        st.markdown(f"*Condition: {doc['metadata'].get('condition')} | Sub-Diagnosis: {doc['metadata'].get('sub_diagnosis')}*")
                        st.text(doc['content'])
                        st.divider()
                
                # Format context for LLM
                context_str = retriever.format_docs(retrieved_docs)

        # 2. Generation Step
        st.subheader("2. Generation Step")
        with st.spinner("Synthesizing answer with Local LLM (this may take a moment)..."):
            answer = rag_chain.generate_answer(query, context_str)
            
        # Display Answer
        st.markdown("### 🤖 Diagnostic Assessment")
        st.markdown(answer)
        
        # 3. Evaluation Step
        st.subheader("3. Evaluation Metrics")
        with st.spinner("Calculating metrics..."):
            metrics = evaluator.calculate_metrics(query, answer, context_str)
            
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Relevance Score (Query vs Answer)", f"{metrics['relevance_score']:.4f}")
        with col2:
            st.metric("Faithfulness Score (Answer vs Context)", f"{metrics['faithfulness_score']:.4f}")
            
        st.caption("Scores range from -1 to 1. Higher is better.")
