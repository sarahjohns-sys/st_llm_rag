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
    # This will print the full path so you can see exactly where it's looking!
    abs_path = os.path.abspath(directory_path)
    print(f"Starting soul-rebuild from: {abs_path}")

    if not os.path.exists(directory_path):
        print(f"❌ ERROR: Cannot find the folder: {abs_path}")
        return

    # Look for JSONL files in the current folder
    for filename in os.listdir(directory_path):
        if filename.endswith(".jsonl"):
            print(f"  ✨ Found identity file: {filename}")
            with open(os.path.join(directory_path, filename), 'r', encoding='utf-8') as f:
                for line in f:
                    data = json.loads(line)
                    text = data.get("content") or data.get("text")
                    if text:
                        # Preservation of your metadata logic for rag_app.py
                        file_meta = data.get("metadata", {})
                        file_meta["source"] = filename 
                        
                        all_docs.append(Document(
                            page_content=text,
                            metadata=file_meta
                        ))

    if all_docs:
        # This saves it to the 'faiss_index' folder that rag_app.py uses
        vectorstore = FAISS.from_documents(all_docs, embeddings)
        vectorstore.save_local("faiss_index")
        print(f"✅ Success! Rebuilt Orrin's soul with {len(all_docs)} fragments.")
    else:
        print("❌ No .jsonl files found. Check your folder!")

# Use "." to look in the current folder
rebuild_index(".")
