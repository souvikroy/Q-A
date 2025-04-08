import streamlit as st
import os
import sys
from pathlib import Path

# Add the project root directory to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.core.document_qa import DocumentChunker, load_document
from src.core.vector_store import VectorStore
import tempfile
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get OpenAI API key from environment
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Set page config
st.set_page_config(
    page_title="Document Analysis AI",
    page_icon="📄",
    layout="wide"
)

# Initialize session state
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'document_loaded' not in st.session_state:
    st.session_state.document_loaded = False

def process_document(uploaded_file):
    """Process the uploaded document and create vector store."""
    # Create a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.txt') as tmp_file:
        # Write uploaded content to temp file
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name

    try:
        with st.status("Processing document...", expanded=True) as status:
            # Load and process document
            status.write("Loading document...")
            document_text = load_document(tmp_path)
            
            status.write("Chunking document...")
            chunker = DocumentChunker()
            chunks = chunker.chunk_document(document_text)

            # Setup vector store
            status.write("Creating vector store...")
            vector_store = VectorStore()
            vector_store.create_vector_store(chunks)
            
            status.write("Setting up QA chain...")
            vector_store.setup_qa_chain()
            
            status.update(label="✅ Document processed successfully!", state="complete")
            return vector_store

    finally:
        # Clean up temp file
        os.unlink(tmp_path)

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
        st.markdown("### 📁 Upload Document")
        uploaded_file = st.file_uploader("Choose a text file", type=['txt', 'pdf'])
        
        if uploaded_file:
            with st.spinner("⚡ Ready to process"):
                if st.button("🚀 Process Document", type="primary", use_container_width=True):
                    st.session_state.vector_store = process_document(uploaded_file)
                    st.session_state.document_loaded = True
                    st.success("✅ Document processed successfully!")

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
    if st.session_state.document_loaded:
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
                            with st.expander(f"Source from {doc.metadata.get('section', 'Document')}"):
                                st.markdown(doc.page_content)
                                st.markdown(f"*Page: {doc.metadata.get('page', 'N/A')}*")
                    
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
