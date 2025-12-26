import os
import json
from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from dotenv import load_dotenv

load_dotenv()

# Initialize Embeddings
embeddings = AzureOpenAIEmbeddings(azure_deployment=os.getenv("AZURE_EMBEDDING_DEPLOYMENT_NAME"))

def rebuild_index(directory_path):
    all_docs = []
    print(f"Starting soul-rebuild from {directory_path}...")

    for filename in os.listdir(directory_path):
        if filename.endswith(".jsonl"):
            print(f"  Ingesting: {filename}")
            with open(os.path.join(directory_path, filename), 'r', encoding='utf-8') as f:
                # Updated loop for rebuild_soul.py
                for line in f:
                    data = json.loads(line)
                    text = data.get("content") or data.get("text")
                    if text:
                        # 1. Grab metadata from the file
                        file_meta = data.get("metadata", {})
                        # 2. FORCE the 'source' key to be the filename
                        # (This ensures it shows up in your Streamlit sources list!)
                        file_meta["source"] = filename 
        
                        all_docs.append(Document(
                            page_content=text,
                            metadata=file_meta
                        ))


    # WIPE AND BUILD
    if all_docs:
        vectorstore = FAISS.from_documents(all_docs, embeddings)
        vectorstore.save_local("faiss_index")
        print(f"✅ Soul Rebuilt! {len(all_docs)} anchors integrated.")

rebuild_index("./llm_rag_test") # Updated to actual folder path
