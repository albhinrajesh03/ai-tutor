from sentence_transformers import SentenceTransformer
import chromadb

model = SentenceTransformer("all-MiniLM-L6-v2")

client = chromadb.PersistentClient(path="./chroma_db")

collection = client.get_or_create_collection(name="pdf_rag")

def split_text(text, chunk_size=500, overlap=100):
    chunks = []
    i = 0
    while i < len(text):
        chunk = text[i:i+chunk_size]
        chunks.append(chunk)
        i = i + chunk_size - overlap
    return chunks

def prepare_chunks(chunks):
    encoded = model.encode(chunks).tolist()
    ids = []
    for i in range(len(chunks)):
        ids.append(str(i))
    collection.add(documents=chunks, embeddings=encoded, ids=ids)

def retrieve(question):
    question_encoded = model.encode(question).tolist()
    results = collection.query(
        query_embeddings=[question_encoded],
        n_results=2
    )
    retrieved_chunks = results["documents"][0]
    return retrieved_chunks