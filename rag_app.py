import streamlit as st
from datetime import datetime

# Core Azure & Vector bits
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document

# THE NEW 2025 PATHS:
# We use 'langchain.memory' but we must ensure the package 'langchain' is version 1.x
from langchain.memory import ConversationSummaryBufferMemory
from langchain.chains import ConversationalRetrievalChain

# --- 0. LONG TERM MEMORY FUNCTION ---
def save_session_summary(llm, vectorstore, chat_history):
    """Summarizes the current chat and saves it permanently to FAISS."""
    try:
        # 1. Format history
        if isinstance(chat_history, list):
            history_str = "\n".join([f"{msg.type}: {msg.content}" for msg in chat_history])
        else:
            history_str = str(chat_history)

        # 2. Prompt for summarization
        summary_prompt = f"""
        Analyze this conversation. Extract high-level insights, decisions, and personal knowledge anchors.
        Focus on evolving knowledge, not simple Q&A.
        
        Conversation:
        {history_str}
        
        Concise Summary (max 500 words):
        """

        # 3. Generate summary
        summary_response = llm.invoke(summary_prompt)
        summary = summary_response.content
        
        if not summary:
            return "No summary generated.", False

        # 4. Create Document
        metadata = {
            "source": "session_summary",
            "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "type": "session_summary",
            "status": "active"
        }
        new_doc = Document(page_content=summary, metadata=metadata)

        # 5. Save to FAISS
        vectorstore.add_documents([new_doc])
        vectorstore.save_local("faiss_index")
        
        return summary, True
    except Exception as e:
        return f"Error: {str(e)}", False

# --- 1. CONFIGURATION ---
EMBEDDING_DEPLOYMENT_NAME = st.secrets["AZURE_EMBEDDING_DEPLOYMENT_NAME"]
CHAT_DEPLOYMENT_NAME = st.secrets["AZURE_CHAT_DEPLOYMENT_NAME"]
API_VERSION = st.secrets["OPENAI_API_VERSION"]

SYSTEM_MESSAGE = (
    "You are a memory-backed conversational assistant. Speak with clarity and compassion. "
    "Reference past logs when appropriate and adapt to the user's style."
)

CUSTOM_PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_MESSAGE),
    ("system", "Context:\n---{context}---"),
    ("human", "{question}")
])

@st.cache_resource
def setup_rag_chain():
    # Load Embeddings
    embeddings = AzureOpenAIEmbeddings(azure_deployment=EMBEDDING_DEPLOYMENT_NAME)

    # Load FAISS
    vectorstore = FAISS.load_local(
        "faiss_index", 
        embeddings, 
        allow_dangerous_deserialization=True
    )
    retriever = vectorstore.as_retriever()

    # Load LLM
    llm = AzureChatOpenAI(
        azure_deployment=CHAT_DEPLOYMENT_NAME,
        model_name="gpt-4o-mini",
        temperature=0.7,
        openai_api_version=API_VERSION
    )

    # Memory Setup
    memory = ConversationSummaryBufferMemory(
        llm=llm,
        max_token_limit=500,
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": CUSTOM_PROMPT}
    )
    
    return qa_chain, llm, vectorstore

# --- 2. INITIALIZATION ---
if 'qa_chain' not in st.session_state:
    st.session_state.qa_chain, st.session_state.llm, st.session_state.vectorstore = setup_rag_chain()
    
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- 3. INTERFACE ---
st.title("Local RAG Chatbot")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask about your history..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.spinner("Thinking..."):
        try:
            result = st.session_state.qa_chain.invoke({"question": prompt})
            response = result['answer']
            
            source_files = set([doc.metadata.get('source') for doc in result.get('source_documents', [])])
            if source_files:
                response += f"\n\n---\n*Sources: {', '.join(source_files)}*"
        except Exception as e:
            response = f"An error occurred: {e}"

    with st.chat_message("assistant"):
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})

# --- 4. SIDEBAR ACTIONS ---
if st.sidebar.button("💾 Save Session to Long-Term Memory"):
    llm = st.session_state.llm
    vectorstore = st.session_state.vectorstore 
    current_chat_history = st.session_state.qa_chain.memory.buffer 
    
    with st.spinner("Saving to FAISS..."):
        summary_text, success = save_session_summary(llm, vectorstore, current_chat_history)
        
    if success:
        st.success("✅ Knowledge saved! Session cleared.")
        st.session_state.messages = []
        st.session_state.qa_chain.memory.clear() # Clear the short-term memory too
    else:
        st.error(summary_text)
