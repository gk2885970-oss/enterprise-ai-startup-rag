from fastapi import FastAPI, Query
import chromadb
from chromadb.utils import embedding_functions

app = FastAPI(title="Indian AI Startup Vector Intelligence API")

# Initialize ChromaDB Client
chroma_client = chromadb.PersistentClient(path="./chroma_db")
ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
collection = chroma_client.get_collection(name="indian_ai_startups", embedding_function=ef)

@app.get("/")
def home():
    return {"message": "Indian AI Startup RAG Engine is Running!"}

@app.get("/search")
def search_startups(query: str = Query(..., description="Natural language search query"), top_k: int = 5):
    """
    Semantic Hybrid Search Endpoint
    Example: /search?query=computer vision startups in video editing&top_k=5
    """
    results = collection.query(
        query_texts=[query],
        n_results=top_k
    )
    
    matches = []
    if results["documents"]:
        for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
            matches.append({
                "company_name": meta["company_name"],
                "city": meta["city"],
                "founded_year": meta["founded_year"],
                "employee_count": meta["employee_count"],
                "special_flags": meta["special_flags"],
                "matched_context": doc
            })
        
    return {
        "query": query,
        "total_results": len(matches),
        "results": matches
    }
