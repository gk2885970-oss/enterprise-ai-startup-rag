import pandas as pd
import chromadb
from chromadb.utils import embedding_functions

print("Starting ChromaDB Indexing Pipeline...")

# 1. Initialize ChromaDB Persistent Client
chroma_client = chromadb.PersistentClient(path="./chroma_db")
ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

collection = chroma_client.get_or_create_collection(
    name="indian_ai_startups",
    embedding_function=ef
)

# 2. Load Tracxn Excel Data
excel_file = "India AI Startup Intelligence Cohort1.xlsx"
df = pd.read_excel(excel_file, sheet_name="Companies Covered 1.1", skiprows=6)
df = df.dropna(subset=["Company Name"])

documents = []
metadatas = []
ids = []

for idx, row in df.iterrows():
    comp_name = str(row["Company Name"]).strip()
    overview = str(row["Overview"]) if pd.notna(row["Overview"]) else ""
    desc = str(row["Description"]) if pd.notna(row["Description"]) else ""
    sector = str(row["Sector (Pratice Area & Feed)"]) if pd.notna(row["Sector (Pratice Area & Feed)"]) else ""
    biz_model = str(row["Business Models"]) if pd.notna(row["Business Models"]) else ""
    
    # Combined context string for Vector Indexing
    text_content = f"Company: {comp_name}. Overview: {overview}. Description: {desc}. Sector: {sector}. Business Model: {biz_model}."
    
    metadata = {
        "company_name": comp_name,
        "founded_year": int(row["Founded Year"]) if pd.notna(row["Founded Year"]) else 0,
        "city": str(row["City"]) if pd.notna(row["City"]) else "Unknown",
        "employee_count": str(row["Total Employee Count"]) if pd.notna(row["Total Employee Count"]) else "N/A",
        "special_flags": str(row["Special Flags: TRUE"]) if pd.notna(row["Special Flags: TRUE"]) else ""
    }
    
    documents.append(text_content)
    metadatas.append(metadata)
    ids.append(f"comp_{idx}")

# Add data into ChromaDB
collection.add(
    documents=documents,
    metadatas=metadatas,
    ids=ids
)

print(f"Successfully indexed {len(documents)} startups into ChromaDB vector store!")