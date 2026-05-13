from sentence_transformers import SentenceTransformer
import faiss

model=SentenceTransformer('all-MiniLM-L6-v2')

def split_text(text, chunk_size=500, overlap=100):
    chunks=[]
    i=0
    while i < len(text):
        chunk=text[i:i+chunk_size]
        chunks.append(chunk)
        i=i+chunk_size-overlap
    return chunks

def prepare_chunks(chunks):
    encoded=model.encode(chunks).astype("float32")
    return encoded

def retrieve(question,chunks,encodings):
    question_encoding=model.encode([question]).astype("float32")

    dimension=question_encoding.shape[1]
    index=faiss.IndexFlatL2(dimension)

    index.add(encodings)

    distance, indices=index.search(question_encoding,2)

    result=[]
    for i in indices[0]:
        result.append(chunks[i])

    return result