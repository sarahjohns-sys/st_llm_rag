import streamlit as st
from datetime import datetime

# Azure bits
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document

# Version 1.x compatible I think?
from langchain_classic.memory import ConversationSummaryBufferMemory
from langchain_classic.chains import ConversationalRetrievalChain

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
    "Reference past logs when appropriate and adapt to the user's style. "
    "PRIORITIZATION RULES: "
    "Some context chunks have a 'status' metadata tag. When information conflicts, you MUST follow this hierarchy: "
    "1. 'active': Highest priority. This is the user's current truth. "
    "2. 'foundational': Core beliefs/data. "
    "3. 'historical': General context, but may be outdated. "
    "4. 'superseded': Lowest priority. Only use if no other info exists."
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
    search_kwargs = {}
    if "status_filter" in st.session_state and st.session_state.status_filter != "All":
        search_kwargs["filter"] = {"status": st.session_state.status_filter.lower()}

retriever = vectorstore.as_retriever(search_kwargs=search_kwargs)

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
st.sidebar.title("Search Controls")
status_options = ["All", "Active", "Foundational", "Historical", "Superseded"]
st.session_state.status_filter = st.sidebar.selectbox("Filter by Status:", status_options)

if st.sidebar.button("Update Filter"):
    # This force-reloads the chain with the new filter
    st.cache_resource.clear()
    st.rerun()

if st.sidebar.button("💾 Save Session to Long-Term Memory"):
    llm = st.session_state.llm
    vectorstore = st.session_state.vectorstore 
    
    # This is the 'raw' history the LLM uses to summarize
    current_chat_history = st.session_state.qa_chain.memory.buffer 
    
    with st.spinner("Saving to FAISS..."):
        summary_text, success = save_session_summary(llm, vectorstore, current_chat_history)
        
    if success:
        st.success("✅ Knowledge saved! Session cleared.")
        
        # 1. Clear the UI list
        st.session_state.messages = [] 
        
        # 2. CLEAR THE RAG CHAIN MEMORY (This is what was missing)
        st.session_state.qa_chain.memory.clear() 
        
        # Optional: Rerun to refresh the UI immediately
        st.rerun()
    else:
        st.error(f"Failed to save: {summary_text}")
