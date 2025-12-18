import os
from dotenv import load_dotenv
from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_community.document_loaders import JSONLoader


# 1. Load environment variables
load_dotenv()

# Get the deployment name for the embedding model from .env
EMBEDDING_DEPLOYMENT_NAME = os.getenv("AZURE_EMBEDDING_DEPLOYMENT_NAME")

# --- A. LOAD ---
print("1. Loading the JSON file...")
# Load the JSON
loader = JSONLoader(
    file_path="anchors_corpus.jsonl",
    jq_schema='.content',  # This grabs the text
    text_content=False,    # Ensures content goes to page_content, not metadata
    metadata_func=lambda x: x["metadata"]
)
documents = loader.load()

# --- B. SPLIT (Chunking) ---
print("2. Splitting documents into chunks...")
# Use a RecursiveCharacterTextSplitter - great for general text
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", " ", ""] # Try to split on paragraphs first, then lines, etc.
)
chunks = text_splitter.split_documents(documents)

# Add custom metadata (the "salient label" practice)
for i, chunk in enumerate(chunks):
    # This metadata will be stored alongside the vector
    chunk.metadata["source"] = "Gemini_Chat_History"
    chunk.metadata["chunk_number"] = i
    chunk.metadata["topic"] = "LLM_RAG_Setup"

print(f"   Created {len(chunks)} chunks for indexing.")

# --- C. EMBED and STORE ---
print("3. Creating embeddings and storing in Chroma...")
# Initialize the Azure OpenAI Embeddings object (this points to your deployed model)
embeddings = AzureOpenAIEmbeddings(
    model=EMBEDDING_DEPLOYMENT_NAME # Uses the deployment name from your .env
)

# Create the Vector Store (this performs the embedding and saves the database locally)
vectorstore = FAISS.from_documents(
    documents=chunks,
    embedding=embeddings
)
# FAISS saves to a single file, not a folder
vectorstore.save_local("faiss_index") 

print("✅ Ingestion Complete! Vector database saved locally to './faiss_index'")
