# Automated Documentation Generation System

A production-ready system for automatically generating comprehensive, software-level documentation for Python codebases using **Gemma model**, **ChromaDB**, **Neo4j**, and **RAG pipeline**.

## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set your Hugging Face token
export HF_TOKEN="your-token-here"

# 3. Run the pipeline
python automate_docs.py --project_path /path/to/your/project
```

## 📋 What It Does (Fully Automated)

The pipeline automatically:

1. **Chunks** your Python code using AST parsing
2. **Embeds** chunks in ChromaDB for semantic search
3. **Generates** call graphs using PyCG and stores in Neo4j
4. **Retrieves** RAG context from CodexGLUE dataset
5. **Generates** detailed documentation using Gemma model
6. **Reassembles** documentation in original file order
7. **Builds** a beautiful Sphinx HTML documentation site

Just provide the project path - everything else is automated!

## 📁 Project Structure

```
auto_doc_project/
├── automate_docs.py              # Main entry point
├── requirements.txt              # Dependencies
├── README.md                    # This file
│
├── src/
│   ├── chunking/
│   │   └── chunk_code.py        # AST-based chunking
│   ├── embedding/
│   │   └── embed_chunks.py      # ChromaDB embedding
│   ├── graph/
│   │   └── generate_call_graph.py  # Neo4j call graphs
│   ├── rag/
│   │   └── rag_context.py       # RAG context retrieval
│   ├── documentation/
│   │   └── generate_docs.py     # Gemma doc generation
│   ├── reassembly/
│   │   └── reassemble_docs.py   # Documentation reassembly
│   ├── web/
│   │   └── sphinx_builder.py   # Sphinx HTML generation
│   └── utils/
│       └── helpers.py           # Utility functions
│
├── chroma_store/                # Generated embeddings
├── rag_chroma/                  # CodexGLUE embeddings
├── neo4j_graph/                 # Generated call graphs
├── generated_docs/               # Generated documentation
└── docs/                        # Sphinx HTML output
    └── _build/html/
```

## 🎯 Usage

### Basic Usage (Fully Automated)

```bash
python automate_docs.py --project_path ./my_project
```

### With Custom Options

```bash
python automate_docs.py \
  --project_path ./my_project \
  --chroma-path ./my_chroma \
  --rag-chroma-path ./my_rag_chroma \
  --neo4j-uri bolt://localhost:7687 \
  --output-dir ./my_docs \
  --docs-dir ./my_sphinx_docs \
  --hf-token your-token
```

## 📊 Pipeline Flow

```
User Input → Chunking → Embedding → Graph → RAG → Documentation → Reassembly → Sphinx → Done!
```

## 🔧 Dependencies

Install all requirements:

```bash
pip install -r requirements.txt
```

Key dependencies:
- `transformers` - Gemma model
- `chromadb` - Vector database
- `sentence-transformers` - Embeddings
- `neo4j` - Graph database
- `sphinx` - HTML documentation
- `pycg` - Call graph generation

## 📝 Output

After running, you'll find:

1. **`./generated_docs/`** - Markdown documentation files
2. **`./docs/_build/html/`** - Beautiful HTML documentation site
3. **`./chroma_store/`** - Vector embeddings database
4. **Neo4j** - Call graphs and relationships

## 🎨 Generated Documentation Includes

- File-level purpose and role
- Function documentation (parameters, returns, logic)
- Code quality evaluation
- Design pattern analysis
- Performance recommendations
- Security analysis
- Testing suggestions
- Dependencies documentation

## ⚙️ Configuration

### Environment Variables

- `HF_TOKEN` - Your Hugging Face token (required)
- `CHROMA_PATH` - ChromaDB storage path (default: `./chroma_store`)
- `NEO4J_PASSWORD` - Neo4j password (default: `test-password`)

### Command Line Arguments

- `--project_path` - Path to your project (required)
- `--chroma-path` - ChromaDB path (optional)
- `--rag-chroma-path` - RAG ChromaDB path (optional)
- `--neo4j-uri` - Neo4j URI (optional)
- `--output-dir` - Output directory (optional)
- `--docs-dir` - Sphinx directory (optional)
- `--hf-token` - Hugging Face token (optional)

## 🐛 Troubleshooting

### Missing Dependencies
```bash
pip install -r requirements.txt
```

### Neo4j Connection Issues
```bash
# Start Neo4j using Docker
docker run --name neo4j -p 7474:7474 -p 7687:7687 neo4j:latest
```

### ChromaDB Errors
```bash
# Clean ChromaDB storage
rm -rf ./chroma_store
```

### Out of Memory
Use a smaller Gemma model or process smaller projects.

## 📚 Documentation

Each generated documentation file includes:

- **Introduction** - File purpose and role
- **Function Documentation** - Parameters, returns, logic
- **Code Quality** - Evaluation and recommendations
- **Design Patterns** - Architecture analysis
- **Performance** - Optimization suggestions
- **Security** - Vulnerability notes
- **Testing** - QA recommendations
- **Dependencies** - Required libraries

## ✨ Features

- ✅ Fully automated pipeline
- ✅ AST-based intelligent chunking
- ✅ Vector embeddings for semantic search
- ✅ Call graph analysis
- ✅ RAG context enhancement
- ✅ Gemma-powered documentation
- ✅ Beautiful Sphinx HTML site
- ✅ Error handling and logging
- ✅ Progress tracking

## 🎉 Ready to Use!

The system is production-ready. Just:

1. Install dependencies
2. Set your Hugging Face token
3. Run on your project
4. View the beautiful HTML documentation!

**That's it! Everything else is automated.**

