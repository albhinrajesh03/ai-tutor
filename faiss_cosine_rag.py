from sentence_transformers import SentenceTransformer
import faiss
import numpy

def split_text(text, chunk_size=500, overlap=100):
    chunks = []
    i = 0
    while i < len(text):
        chunk = text[i:i+chunk_size]
        chunks.append(chunk)         
        i = i + chunk_size - overlap           
    return chunks

model=SentenceTransformer('all-MiniLM-L6-v2')

def prepare_chunks(chunks):
  encoded=model.encode(chunks)
  encoded=encoded/numpy.linalg.norm(encoded, axis=1, keepdims=True)
  encoded=encoded.astype("float32")
  return encoded

def retrieve(question, chunks, encoded):
  question_encoded=model.encode([question])
  question_encoded=question_encoded/numpy.linalg.norm(question_encoded, axis=1, keepdims=True)
  question_encoded=question_encoded.astype("float32")

  dimension=question_encoded.shape[1]
  index=faiss.IndexFlatIP(dimension)
  index.add(encoded)

  scores, indices=index.search(question_encoded,2)

  if scores[0][0]<0.5:
    return []

  result=[]
  for i in indices[0]:
    result.append(chunks[i])

  return result