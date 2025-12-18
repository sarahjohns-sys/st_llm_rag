import os
import argparse
from dotenv import load_dotenv
from langchain_openai import AzureOpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import JSONLoader

# 1. Load environment variables
load_dotenv()

EMBEDDING_DEPLOYMENT_NAME = os.getenv("AZURE_EMBEDDING_DEPLOYMENT_NAME")

# 2. Setup Argument Parser for dynamic file paths
parser = argparse.ArgumentParser(description="Ingest a JSONL file into the vector store.")
parser.add_argument("file_path", help="The path to the .jsonl file you want to ingest")
args = parser.parse_args()

# --- A. LOAD ---
print(f"1. Loading the file: {args.file_path}...")

# This version grabs EVERYTHING inside your "metadata" block
def extract_metadata(record, metadata):
    # 'record' is the full JSON line
    # 'metadata' is the default LangChain metadata (like source file)
    
    # Get the inner metadata dictionary from your file
    inner_meta = record.get("metadata", {})
    
    # Merge them together
    metadata.update(inner_meta)
    return metadata

# Fixed metadata_func to accept TWO arguments (record, metadata) to stop the TypeError
# Also uses args.file_path so it's not hard-coded
loader = JSONLoader(
    file_path=args.file_path,
    jq_schema='.',         # Look at the whole line
    content_key="content", # Use the "content" field for the text
    json_lines=True,
    metadata_func=extract_metadata
)

documents = loader.load()

# --- B. SPLIT (Chunking) ---
print("2. Splitting documents into chunks...")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", " ", ""]
)
chunks = text_splitter.split_documents(documents)

# Add custom metadata tags
for i, chunk in enumerate(chunks):
    chunk.metadata["source_file"] = args.file_path
    chunk.metadata["ingest_date"] = datetime.now() # Or use datetime.now()

print(f"   Created {len(chunks)} chunks for indexing.")

# --- C. EMBED and STORE ---
print("3. Creating embeddings and storing in FAISS...")
embeddings = AzureOpenAIEmbeddings(
    azure_deployment=EMBEDDING_DEPLOYMENT_NAME 
)

# If an index already exists, we load it and add to it; otherwise, create new
if os.path.exists("faiss_index"):
    print("   Existing index found. Adding to it...")
    existing_vs = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    existing_vs.add_documents(chunks)
    existing_vs.save_local("faiss_index")
else:
    print("   No index found. Creating new one...")
    vectorstore = FAISS.from_documents(documents=chunks, embedding=embeddings)
    vectorstore.save_local("faiss_index")

print("✅ Ingestion Complete! Vector database updated at './faiss_index'")
