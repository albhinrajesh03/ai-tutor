from sentence_transformers import SentenceTransformer
from sentence_transformers import CrossEncoder
import chromadb

model=SentenceTransformer('all-MiniLM-L6-v2')
reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
client=chromadb.PersistentClient(path="./chroma_db")
collection=client.get_or_create_collection(name="pdf_rag")

def split_text(text, chunk_size=500, overlap=100):
  chunks=[]
  i=0
  while i<(len(text)):
    chunk=text[i:i+chunk_size]
    chunks.append(chunk)
    i=i+chunk_size-overlap
  return chunks

def prepare_chunks(chunks):
  encoded=model.encode(chunks).tolist()
  ids=[]
  for i in range(len(chunks)):
    ids.append(str(i))
  collection.add(documents=chunks, embeddings=encoded, ids=ids)

def retrieve(question):
  encoded_question=model.encode([question]).tolist()
  results=collection.query(query_embeddings=encoded_question, n_results=5)
  retrieved_chunks=results["documents"][0]
  
  pairs=[]
  for chunk in retrieved_chunks:
    pairs.append([question,chunk])
  scores=reranker.predict(pairs)
  ranked=list(zip(retrieved_chunks, scores))
  ranked.sort(key=lambda x:x[1], reverse=True)
  final_chunks=[]
  for chunk, score in ranked[:2]:
    if score > 0.5:
      final_chunks.append(chunk)
  return final_chunks