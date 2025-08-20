import streamlit as st
from agent import RAGAgent

# Configure the page
st.set_page_config(
    page_title="RAG Q&A Assistant", 
    page_icon="🤖", 
    layout="wide"
)

# Initialize session state
if 'agent' not in st.session_state:
    st.session_state.agent = None
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Header
st.title("🤖 RAG Q&A Assistant")
st.markdown("Ask questions about the loaded documents and get AI-powered answers with citations.")

# Sidebar for configuration
with st.sidebar:
    st.header("Configuration")
    
    # Document path configuration
    document_paths = st.text_area(
        "Document Paths (one per line):",
        value="neurolink-system.txt",
        help="Enter the paths to your documents, one per line"
    )
    
    # Advanced settings
    with st.expander("Advanced Settings"):
        chunk_size = st.slider("Chunk Size", 100, 1000, 500)
        chunk_overlap = st.slider("Chunk Overlap", 0, 200, 50)
        k_docs = st.slider("Number of Retrieved Documents", 1, 10, 2)
    
    # Initialize agent button
    if st.button("Initialize Agent", type="primary"):
        try:
            doc_paths = [path.strip() for path in document_paths.split('\n') if path.strip()]
            st.session_state.agent = RAGAgent(
                document_paths=doc_paths,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                k=k_docs
            )
            st.success("Agent initialized successfully!")
        except Exception as e:
            st.error(f"Error initializing agent: {str(e)}")

# Main chat interface
if st.session_state.agent:
    st.success("✅ Agent is ready!")
    
    # Display current hyperparameters
    with st.expander("Current Hyperparameters"):
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Chunk Size", st.session_state.agent.chunk_size)
            st.metric("K (Retrieval)", st.session_state.agent.k)
        with col2:
            st.metric("Chunk Overlap", st.session_state.agent.chunk_overlap)
            st.metric("Embedding Model", type(st.session_state.agent.embedding_model).__name__)
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "assistant":
                # Display answer
                st.markdown(f"**Answer:** {message['content']['answer']}")
                
                # Display citations if available
                if message['content'].get('citations'):
                    st.markdown("**Citations:**")
                    for i, citation in enumerate(message['content']['citations'], 1):
                        st.markdown(f"{i}. {citation}")
                
                # Display retrieved document references if available
                if message['content'].get('retrieved_docs'):
                    st.markdown("**Retrieved Document References:**")
                    for i, doc in enumerate(message['content']['retrieved_docs'], 1):
                        preview = doc[:30] + "..." if len(doc) > 30 else doc
                        with st.expander(f"Reference {i}: {preview}"):
                            st.text(doc)
            else:
                st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask a question about your documents..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get response from agent
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # Get retrieved documents first
                    retrieved_docs = st.session_state.agent.retrieve(prompt)
                    response = st.session_state.agent.answer(prompt)
                    
                    if "error" in response:
                        st.error(f"Error: {response['error']}")
                        if "raw_output" in response:
                            st.code(response['raw_output'])
                    else:
                        # Display answer
                        st.markdown(f"**Answer:** {response['answer']}")
                        
                        # Display citations if available
                        if response.get('citations'):
                            st.markdown("**Citations:**")
                            for i, citation in enumerate(response['citations'], 1):
                                st.markdown(f"{i}. {citation}")
                        
                        # Display retrieved document references
                        if retrieved_docs:
                            st.markdown("**Retrieved Document References:**")
                            for i, doc in enumerate(retrieved_docs, 1):
                                preview = doc[:30] + "..." if len(doc) > 30 else doc
                                with st.expander(f"Reference {i}: {preview}"):
                                    st.text(doc)
                        
                        # Add assistant response to chat history with retrieved docs
                        st.session_state.messages.append({
                            "role": "assistant", 
                            "content": {**response, "retrieved_docs": retrieved_docs}
                        })
                
                except Exception as e:
                    st.error(f"Error getting response: {str(e)}")

else:
    st.warning("⚠️ Please initialize the agent first using the sidebar.")
    st.markdown("""
    ### How to use:
    1. Configure your document paths in the sidebar
    2. Adjust advanced settings if needed
    3. Click "Initialize Agent" 
    4. Start asking questions!
    """)

# Footer
st.markdown("---")
st.markdown("Built with Streamlit and LangChain")