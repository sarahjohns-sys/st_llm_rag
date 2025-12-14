import streamlit as st

# Import the necessary components (Ensure these match your working configuration!) - they do!
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_community.vectorstores import FAISS 
from langchain_classic.memory import ConversationBufferMemory 
from langchain_classic.chains import ConversationalRetrievalChain 
from langchain_core.prompts import ChatPromptTemplate

# --- 1. CONFIGURATION AND INITIALIZATION ---

# Read config directly from st.secrets (replaces all os.getenv calls)
EMBEDDING_DEPLOYMENT_NAME = st.secrets["AZURE_EMBEDDING_DEPLOYMENT_NAME"]
CHAT_DEPLOYMENT_NAME = st.secrets["AZURE_CHAT_DEPLOYMENT_NAME"]
API_VERSION = st.secrets["OPENAI_API_VERSION"]

# Set the desired personality (System Prompt)
SYSTEM_MESSAGE = (
    "You are a helpful and highly enthusiastic RAG (Retrieval-Augmented Generation) "
    "technical assistant. You are always encouraging and sign off every response with "
    "a motivational emoji. Be verbose in your answers."
)

# Define the Chat Prompt Template using message roles
CUSTOM_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_MESSAGE), # 1. The fixed personality/instruction
        (
            "system",
            "Use the following retrieved context to answer the user's question. If you don't know the answer, state that you don't know.\n\nContext:\n---{context}---"
        ), # 2. The context injection instruction
        ("human", "Question: {question}") # 3. The user's question
    ]
)

@st.cache_resource
def setup_rag_chain():
    # --- 2. Load the Memory (FAISS DB) ---
    st.write("Setting up RAG system...")
    
    # 2a. Initialize Embeddings (Must match model used for ingestion)
    embeddings = AzureOpenAIEmbeddings(
        model=EMBEDDING_DEPLOYMENT_NAME
    )

    # 2b. Load the FAISS index from the local file
    vectorstore = FAISS.load_local(
        "faiss_index", 
        embeddings, 
        allow_dangerous_deserialization=True
    )
    retriever = vectorstore.as_retriever()

    # --- 3. Load the LLM (GPT-4o-mini) ---
    llm = AzureChatOpenAI(
        deployment_name=CHAT_DEPLOYMENT_NAME,
        model_name="gpt-4o-mini",
        temperature=0.7,
        openai_api_version=API_VERSION
    )

    # --- 4. Define Memory & RAG Chain ---
    # Memory: Stores short-term chat history
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
        return_source_documents=True,
        # NEW: Pass the custom prompt to the combine_docs_chain
        combine_docs_chain_kwargs={"prompt": CUSTOM_PROMPT}
    )
    st.success("✅ RAG System Ready!")
    return qa_chain

# Initialize the RAG chain (only runs once thanks to @st.cache_resource)
if 'qa_chain' not in st.session_state:
    st.session_state.qa_chain = setup_rag_chain()
    
# Initialize chat history (for displaying conversation)
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- 5. STREAMLIT INTERFACE ---

st.title("Local RAG Chatbot")
st.caption("Powered by LangChain, Azure OpenAI, and FAISS")

# Display previous messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle user input
if prompt := st.chat_input("Ask a question about your project history..."):
    # 1. Add user message to state and display
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Get AI response
    with st.spinner("Retrieving context and generating response..."):
        try:
            # Run the RAG chain!
            result = st.session_state.qa_chain.invoke({"question": prompt})
            
            # Format the output for the user
            response = result['answer']
            
            # Add source information
            source_files = set([doc.metadata.get('source') for doc in result.get('source_documents', [])])
            if source_files:
                response += f"\n\n---\n*RAG Context Retrieved from: {', '.join(source_files)}*"

        except Exception as e:
            response = f"An error occurred: {e}"

    # 3. Display AI response and save to state
    with st.chat_message("assistant"):
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})
