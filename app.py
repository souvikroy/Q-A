import streamlit as st
from streamlit.runtime.uploaded_file_manager import UploadedFile
import os
from document_qa import DocumentChunker, load_document
from vector_store import VectorStore
import tempfile
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document  # Import LangChain's Document class

# Load environment variables
load_dotenv()

# Get OpenAI API key from environment
# Set page config
st.set_page_config(
    page_title="Document Analysis AI",
    page_icon="📄",
    layout="wide"
)

# Initialize session state
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'documents_loaded' not in st.session_state:
    st.session_state.documents_loaded = False
    

def process_documents(uploaded_files: list[UploadedFile]):
    """Process multiple uploaded PDF documents and create a combined vector store."""
    temp_paths = []

    try:
        with st.status("Processing documents...", expanded=True) as status:
            all_chunks = []
            chunker = DocumentChunker()  # Initialize the custom DocumentChunker

            for uploaded_file in uploaded_files:
                # Create a temporary file for each uploaded PDF file
                with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    temp_paths.append(tmp_file.name)

                # Load and process each PDF document
                status.write(f"Loading document: {uploaded_file.name}...")
                loader = PyPDFLoader(temp_paths[-1])
                documents = loader.load()

                # Extract text from the documents
                for doc in documents:
                    # Check if document_text is empty
                    if not doc.page_content.strip():
                        status.write(f"⚠️ No text found in {uploaded_file.name}. Skipping...")
                        continue

                    # Chunk the document using the custom DocumentChunker
                    status.write(f"Chunking document: {uploaded_file.name}... {doc.metadata}")
                    chunks = chunker.chunk_document(text=doc.page_content, document_title=uploaded_file.name, metadata=doc.metadata)

                    # Convert chunks (dicts) into LangChain Document objects
                    for chunk in chunks:
                        all_chunks.append(Document(page_content=chunk["content"], metadata=chunk["metadata"]))

            # Setup vector store for all chunks
            status.write("Creating combined vector store...")
            vector_store = VectorStore()
            vector_store.create_vector_store(all_chunks)

            status.write("Setting up QA chain...")
            vector_store.setup_qa_chain()

            status.update(label="✅ All documents processed successfully!", state="complete")
            return vector_store

    finally:
        # Clean up all temp files
        for path in temp_paths:
            os.unlink(path)
def main():
    # Custom CSS for better styling
    st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
    }
    .css-1v0mbdj.e115fcil1 {
        max-width: 1200px;
        margin: auto;
    }
    .source-box {
        border: 1px solid #e0e0e0;
        border-radius: 5px;
        padding: 1rem;
        margin: 0.5rem 0;
        background-color: #f8f9fa;
    }
    </style>
    """, unsafe_allow_html=True)

    # Header with custom styling
    st.markdown("""
    <h1 style='text-align: center; color: #1E88E5;'>📄 Document Analysis AI</h1>
    <p style='text-align: center; font-size: 1.2em;'>Upload your document and ask questions about its contents.</p>
    """, unsafe_allow_html=True)

    # Sidebar with improved styling
    with st.sidebar:
        st.markdown("### 📁 Upload Documents")
        uploaded_files = st.file_uploader("Choose PDF files", type=['pdf'], accept_multiple_files=True)

        if uploaded_files:
            with st.spinner("⚡ Ready to process"):
                if st.button("🚀 Process Documents", type="primary", use_container_width=True):
                    st.session_state.vector_store = process_documents(uploaded_files)
                    st.session_state.documents_loaded = True
                    st.success("✅ Documents processed successfully!")

        st.divider()
        st.markdown("""
        ### 💡 Sample Questions
        1. What are the technical qualification requirements?
        2. What are the financial qualification requirements?
        3. What are the criteria for a Joint Venture (JV)?
        4. What is the minimum turnover requirement?
        5. What experience is required for key personnel?
        """)
        
        # Add helpful tips
        with st.expander("ℹ️ Tips for better results"):
            st.markdown("""
            - Be specific in your questions
            - Use keywords from the document
            - Ask one question at a time
            - Include relevant context in your question
            """)

    # Main content with improved layout
    if st.session_state.documents_loaded:
        # Question input with improved styling
        st.markdown("### 🤔 Ask Your Question")
        question = st.text_input(
            "Type your question here:",
            placeholder="e.g., What are the technical qualification requirements?",
            key="question_input"
        )
        
        if question:
            with st.spinner("🔍 Finding answer..."):
                try:
                    # Use qa_chain instead of direct query
                    result = st.session_state.vector_store.qa_chain({"query": question})
                    
                    # Display answer in a nice container
                    st.markdown("### 📝 Answer")
                    st.markdown(result["result"])
                    
                    # Display sources if available
                    if "source_documents" in result:
                        st.markdown("### 📚 Sources")
                        for doc in result["source_documents"]:
                            print(doc.metadata)
                            with st.expander(f"Source from {doc.metadata.get('section', 'Document')}"):
                                st.markdown(doc.page_content)
                                st.markdown(f"*Page: {  doc.metadata.get('page', 'N/A')}*")
                                st.markdown(f"*Document: {doc.metadata.get('title', 'N/A')}*")
                    
                except Exception as e:
                    st.error(f"Error getting answer: {str(e)}")
                    st.markdown("Please try rephrasing your question or uploading the document again.")
    else:
        # Display instructions with improved styling
        st.info("👈 Please upload a document and click 'Process Document' to begin.")
        
        # Add feature highlights in a grid
        st.markdown("### ✨ Features & Capabilities")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            #### 📊 Document Processing
            - Smart text chunking
            - Advanced OCR support
            - Multiple file formats
            - Efficient processing
            """)
            
        with col2:
            st.markdown("""
            #### 🤖 AI Analysis
            - Context-aware QA
            - Source citations
            - Semantic search
            - Natural language
            """)
            
        with col3:
            st.markdown("""
            #### 📈 Use Cases
            - Technical documents
            - Legal contracts
            - Financial reports
            - Project specs
            """)

if __name__ == "__main__":
    main()
