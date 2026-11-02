# Enterprise AI Startup Discovery & Intelligence System (GenAI + Vector RAG)

An enterprise-grade **Retrieval-Augmented Generation (RAG)** system designed to perform high-precision semantic discovery and intelligence retrieval across 40+ Indian AI startups using real-world Tracxn cohort data.

- 🔗 **Live Demo**: [Enterprise AI Startup RAG Engine](https://enterprise-ai-startup-rag-xuqbw5ewarb7atzaqhuktj.streamlit.app)
- 🐙 **GitHub Repository**: [enterprise-ai-startup-rag](https://github.com/gk2885970-oss/enterprise-ai-startup-rag)

---

##  Key Features
- **Semantic Vector Search**: Generates dense vector embeddings using `all-MiniLM-L6-v2` to query unstructured company overviews, technical taxonomies, and business models.
- **Persistent Vector Database**: Embeddings stored in **ChromaDB** with structured metadata indexing (City, Founded Year, Employee Count, Special Flags).
- **Interactive UI**: Built with Streamlit featuring dynamic result expansion, similarity controls, and context inspection.

---

##  Tech Stack
- **Language**: Python 3.11+
- **Vector Database**: ChromaDB
- **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2`
- **Frontend / Cloud Host**: Streamlit Community Cloud

---

## 📂 Project Structure
```text
├── chroma_db/                  # Persistent Vector Database Index
├── India AI Startup Intelligence Cohort1.xlsx  # Tracxn Intelligence Dataset
├── ingest.py                   # ChromaDB Data Ingestion & Embedding Pipeline
├── main.py                     # FastAPI Retrieval Microservice Backend
├── app.py                      # Production Streamlit RAG Application
├── requirements.txt            # Project Dependencies
└── README.md                   # System Documentation
