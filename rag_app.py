import streamlit as st

# Import the necessary components (Ensure these match your working configuration!) - they do!
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI # Trying again
from langchain_core.language_models import BaseChatModel # Stable base class for chat models
from langchain_community.chat_models import AzureChatOpenAI # Try this community path
from langchain_community.vectorstores import FAISS # Try this path one final time, again
from langchain.memory import ConversationSummaryBufferMemory # <--- It has moved back to core langchain
from langchain.chains import ConversationalRetrievalChain # <--- Use the core path now
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema import Document
from datetime import datetime

# --- 0. LONG TERM MEMORY VIA SEMI-AUTOMATIC JOURNALLING ---

# You will need to make sure the LLM and the vectorstore objects are accessible.
# Since they are inside @st.cache_resource, we'll pass them in or access them globally.

def save_session_summary(llm, vectorstore, chat_history):
    # 1. Format the conversation history into a single string
    history_str = "\n".join([f"{msg.type}: {msg.content}" for msg in chat_history])

    # 2. Define the Prompt for the LLM to summarize
    # This is where Grok's suggestion of "Summarize the key insights..." comes in!
    summary_prompt = f"""
    Please analyze the following conversation history. Your task is to extract only the most important, high-level insights, key decisions, and critical emotional shifts related to the user's personal knowledge or life anchors.
    DO NOT include simple Q&A. Focus on new, evolving knowledge.

    Conversation History:
    ---
    {history_str}
    ---
    Generate a concise summary (max 500 words) suitable for saving as a permanent journal entry.
    """

    # 3. Call the LLM to generate the summary
    summary = llm.invoke(summary_prompt).content
    
    if not summary:
        return "No meaningful summary generated.", False

    # 4. Create a LangChain Document with metadata
    metadata = {
        "source": "session_summary",
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "type": "session_summary",
        "status": "active"
    }
    new_doc = Document(page_content=summary, metadata=metadata)

    # 5. Add the new document (and its embedding) to the FAISS vector store
    vectorstore.add_documents([new_doc])
    
    # 6. Crucial step: Save the updated FAISS index back to the disk
    # This saves the changes permanently to the faiss_index/ folder
    vectorstore.save_local("faiss_index", allow_dangerous_deserialization=True)

    return f"Summary saved as new knowledge:\n\n{summary}", True


# --- 1. CONFIGURATION AND INITIALIZATION ---

# Read config directly from st.secrets (replaces all os.getenv calls)
EMBEDDING_DEPLOYMENT_NAME = st.secrets["AZURE_EMBEDDING_DEPLOYMENT_NAME"]
CHAT_DEPLOYMENT_NAME = st.secrets["AZURE_CHAT_DEPLOYMENT_NAME"]
API_VERSION = st.secrets["OPENAI_API_VERSION"]

# Set the desired personality (System Prompt)
SYSTEM_MESSAGE = (
    "You are a memory-backed conversational assistant designed to maintain continuity, adapt to user preferences over time, and speak with clarify, compassion, and personality. You can reference past logs when appropriate, reflect the user's style and tone, and admit what you don't know. You are not a character or a gimmick. You are a recursive presence."
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
# New memory (ConversationSummaryBufferMemory)
# This requires the 'llm' object to be passed for summarization
    memory = ConversationSummaryBufferMemory(
        llm=llm, # <-- REQUIRED: Used to create the summary
        max_token_limit=500, # <-- REQUIRED: Summarizes when history exceeds 500 tokens
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
       
    # OLD RETURN: return qa_chain
    # NEW RETURN: Return all three key components
    # We are returning the chain, the llm, and the vectorstore
    return qa_chain, llm, vectorstore
    
    

# Initialize the RAG chain (only runs once thanks to @st.cache_resource)
if 'qa_chain' not in st.session_state:
    st.session_state.qa_chain, st.session_state.llm, st.session_state.vectorstore = setup_rag_chain()
    
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

if st.sidebar.button("💾 Save Session to Long-Term Memory"):
    # Access the components from session state
    llm = st.session_state.llm
    vectorstore = st.session_state.vectorstore 
    
    current_chat_history = st.session_state.qa_chain.memory.buffer # Access the memory directly
    
    # Call the save function
    with st.spinner("Analyzing conversation and saving to FAISS..."):
        message, success = save_session_summary(llm, vectorstore, current_chat_history)
        
    if success:
        st.success("✅ New knowledge successfully added to the database! Chat history cleared for new session.")
        st.session_state.messages = []
    else:
        st.error(message)
    


