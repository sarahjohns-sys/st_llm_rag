import os
from dotenv import load_dotenv

from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_community.vectorstores import FAISS # New vector store
from langchain_classic.memory import ConversationBufferMemory
from langchain_classic.chains import ConversationalRetrievalChain
API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")

# 1. Load environment variables
load_dotenv()

# --- Load Configuration ---
# Your deployment names (from .env)
EMBEDDING_DEPLOYMENT_NAME = os.getenv("AZURE_EMBEDDING_DEPLOYMENT_NAME")
CHAT_DEPLOYMENT_NAME = os.getenv("AZURE_CHAT_DEPLOYMENT_NAME")

# --- 2. Load the Memory (Chroma DB) ---
print("1. Loading Vector Database (Long-Term Memory)...")
# Initialize the Azure OpenAI Embeddings object (needs to match the model used for ingestion)
embeddings = AzureOpenAIEmbeddings(
    model=EMBEDDING_DEPLOYMENT_NAME
)

# Load the existing vectorstore from the 'chroma_db' folder
# Replace with this FAISS loading logic:
# Tell LangChain you trust this file since you created it yourself
vectorstore = FAISS.load_local(
    "faiss_index", 
    embeddings, 
    allow_dangerous_deserialization=True # <-- ADD THIS LINE
)

# Create a Retriever from the vector store
retriever = vectorstore.as_retriever()

# --- 3. Load the LLM (GPT-3.5-Turbo) ---
print("2. Loading Azure Chat Model (The Intelligence)...")
llm = AzureChatOpenAI(
    deployment_name=CHAT_DEPLOYMENT_NAME,
    model_name="gpt-4o-mini", # This should match the model type of your deployment
    temperature=0.7,
    openai_api_version=API_VERSION
)

# --- 4. Define Memory & RAG Chain ---
print("3. Setting up Conversational RAG Chain...")

# Short-Term Memory: Stores the last few interactions as raw text
memory = ConversationBufferMemory(
    memory_key="chat_history", 
    return_messages=True,
    output_key="answer"
)

# The ConversationalRetrievalChain combines the LLM, the Retriever, and the Memory
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory,
    return_source_documents=True # Get the source document chunks used to generate the answer
)

# --- 5. Start the Chat Loop ---
print("\n✅ RAG System Ready! Start chatting. Type 'quit' to exit.")
print("---")

while True:
    query = input("You: ")
    if query.lower() == 'quit':
        break
    
    # Run the chain! It automatically manages memory and retrieval
    result = qa_chain.invoke({"question": query})

    print(f"\nAssistant: {result['answer']}")
    
    # Optional: show the source documents for verification (proof of RAG)
    source_files = set([doc.metadata.get('source') for doc in result.get('source_documents', [])])
    if source_files:
        print(f"\n[RAG Source Used: {', '.join(source_files)}]")
