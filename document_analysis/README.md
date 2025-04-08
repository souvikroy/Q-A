# Document Analysis AI

An intelligent document analysis system specialized in processing tender documents, RFPs, and technical specifications. The system uses advanced NLP and vector search to provide accurate answers to specific queries about document contents.

## Features

- 📄 Smart document processing with hierarchical chunking
- 🔍 Advanced semantic search using FAISS
- 💡 Context-aware question answering
- 📊 Source citations and reference tracking
- 🚀 Modern web interface with Streamlit

## Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd document_analysis
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

4. Set up environment variables:
Create a `.env` file in the project root:
```env
OPENAI_API_KEY=your_api_key_here
```

## Usage

1. Start the web application:
```bash
streamlit run src/web/app.py
```

2. Upload a document through the web interface

3. Ask questions about the document contents

## Project Structure

```
document_analysis/
├── src/                    # Source code
│   ├── core/              # Core functionality
│   ├── api/               # API endpoints
│   ├── utils/             # Utility functions
│   └── web/               # Web interface
├── config/                # Configuration files
├── data/                  # Data files
├── docs/                  # Documentation
└── tests/                 # Test files
```

## Development

1. Run tests:
```bash
python -m pytest
```

2. Format code:
```bash
black .
isort .
```

3. Check code quality:
```bash
flake8
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- LangChain for the excellent LLM framework
- OpenAI for the powerful language models
- FAISS for efficient vector similarity search
- Streamlit for the amazing web framework
