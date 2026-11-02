import streamlit as st
import chromadb
from chromadb.utils import embedding_functions

st.set_page_config(page_title="Indian AI Startup Discovery Engine", layout="wide")

st.title("🤖 Enterprise AI Startup Intelligence Engine (RAG)")
st.caption("Powered by ChromaDB Vector Search & Tracxn Dataset")

# Cache vector client to avoid re-loading on every interaction
@st.cache_resource
def get_vector_collection():
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
    return chroma_client.get_collection(name="indian_ai_startups", embedding_function=ef)

try:
    collection = get_vector_collection()
except Exception as e:
    st.error(f"Vector Database load karne me issue aaya. Pehle `python ingest.py` run karke `./chroma_db` generate karein. Error: {e}")
    st.stop()

# Sidebar Settings
st.sidebar.header("Search Settings")
top_k = st.sidebar.slider("Number of Top Matches", min_value=1, max_value=10, value=5)

# Query Input
user_query = st.text_input(
    "Enter natural language query to discover startups:",
    placeholder="e.g., Show me AI startups building automated video creation tools or voice NLP models"
)

if user_query:
    with st.spinner("Searching Vector Database..."):
        results = collection.query(query_texts=[user_query], n_results=top_k)
        
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
        
        st.subheader(f"Top {len(matches)} Semantic Matches for: '{user_query}'")
        
        if matches:
            for idx, item in enumerate(matches, 1):
                with st.expander(f"🏢 {idx}. **{item['company_name']}** ({item['city']} | Founded: {item['founded_year']})", expanded=(idx==1)):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(f"**Team Size**: {item['employee_count']}")
                        st.write(f"**Special Flags**: {item['special_flags']}")
                    with col2:
                        st.write(f"**Location**: {item['city']}")
                    
                    st.markdown("---")
                    st.write("**Retrieved Vector Context:**")
                    st.info(item['matched_context'])
        else:
            st.warning("No relevant startups found matching your query.")