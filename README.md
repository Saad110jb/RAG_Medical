# RAG_Medical
This project implements an end-to-end RAG pipeline designed to answer clinical questions and generate context-aware summaries using the MIMIC-IV-Ext Direct dataset.

# DIRECT: Diagnostic Reasoning RAG System

**DIRECT** (Diagnostic Reasoning for Clinical Notes) is a Retrieval-Augmented Generation (RAG) system designed to assist in medical diagnostics. It retrieves relevant clinical notes from the MIMIC-IV-Ext dataset and uses Google's Gemini LLM to synthesize diagnostic answers.

## 🚀 Features

- **Ingestion**: Parses nested clinical JSON data into a structured format.
- **Vector Store**: Uses **ChromaDB** with `sentence-transformers` for efficient semantic search.
- **Retrieval**: **LangChain**-based retriever to find the most relevant patient history.
- **Generation**: **Gemini 1.5 Flash** (via LangChain) to generate evidence-based diagnoses.
- **User Interface**: Interactive **Streamlit** app with visibility into the retrieval process.

## 📂 Project Structure

```
Clinical-RAG-Direct/
├── .env                    # API Keys (Gemini)
├── requirements.txt        # Python dependencies
├── README.md               # Project documentation
├── app.py                  # Streamlit User Interface
├── src/
│   ├── ingestion.py        # Data parsing logic
│   ├── vector_store.py     # ChromaDB setup
│   ├── retriever.py        # LangChain Retriever
│   └── generator.py        # LangChain RAG Chain
└── data/
    ├── raw/                # Clinical JSON files
    └── chroma_db/          # Persisted Vector Store
```

## 🛠️ Setup & Installation

1.  **Clone/Open the Project**:
    Ensure you are in the `Clinical-RAG-Direct` directory.

2.  **Install Dependencies**:

    ```bash
    pip install -r requirements.txt
    ```

3.  **Set API Key**:
    Create a `.env` file in the root directory:

    ```
    GEMINI_API_KEY=your_google_api_key
    ```

4.  **Ingest Data** (First Run Only):
    If you haven't populated the database yet, run:
    ```bash
    python src/vector_store.py
    ```
    _Note: This will read from `data/raw` or the `mimic-iv-ext-direct` folder._

## 🖥️ Running the Application

Run the Streamlit app:

```bash
streamlit run app.py
```

## 🧠 How It Works

1.  **User Query**: You ask a question (e.g., "Why does the patient have hypoxia?").
2.  **Retrieval**: The system searches ChromaDB for the top _k_ most similar clinical notes.
3.  **Display**: The UI shows the retrieved notes, including metadata (Condition, Sub-Diagnosis).
4.  **Generation**: The retrieved context + query are sent to Gemini, which generates a diagnostic answer citing the evidence.

## ⚠️ Disclaimer

This tool is for **educational and research purposes only**. It is not intended for actual clinical diagnosis or decision-making.
