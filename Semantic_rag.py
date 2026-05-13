from sentence_transformers import SentenceTransformer
import numpy

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
    encoded=model.encode(chunks)
    return encoded

def retrieve(question,chunks,encodings):
    question_encoding=model.encode(question)

    similarities=[]
    for emb,chunk in zip(encodings,chunks):
        score=numpy.dot(emb,question_encoding)/(numpy.linalg.norm(emb)*numpy.linalg.norm(question_encoding))
        similarities.append((score,chunk))

    result=[]

    similarities.sort(reverse=True, key=lambda x:x[0])

    if not similarities:
        return []

    if similarities[0][0]<0.3:
        return []

    top_k=3
    for score,chunk in similarities[:top_k]:
        result.append(chunk)

    return result