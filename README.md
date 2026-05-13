AI Tutor

A simple semantic RAG-based AI Tutor built using:
- Python
- Ollama (`phi3:mini`)
- Sentence Transformers (`all-MiniLM-L6-v2`)

Features:
- PDF text extraction
- Text chunking
- Semantic search using embeddings
- Cosine similarity retrieval
- Local LLM integration
- Context-aware question answering

How It Works:
1. PDF text is extracted
2. Text is split into chunks
3. Each chunk is converted into embeddings
4. User question is encoded
5. Cosine similarity finds the most relevant chunks
6. Relevant context is sent to the LLM

Run Project:
python app.py