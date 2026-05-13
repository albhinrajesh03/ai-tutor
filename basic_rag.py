def split_text(text, chunk_size=500):
    chunks = []
    i = 0
    while i < len(text):
        chunk = text[i:i+chunk_size]
        chunks.append(chunk)         
        i = i + chunk_size            
    return chunks

def retrieve(question, chunks):
    question_words = question.lower().split()
    scored = []

    for chunk in chunks:
        chunk_lower = chunk.lower()
        score = 0
        for word in question_words:
            if word in chunk_lower:
                score = score + 1
        if score > 0:
            scored.append([score, chunk])

    scored.sort(reverse=True,key=lambda x:x[0])

    top_chunks = []
    for item in scored[:3]:
        top_chunks.append(item[1])
    return top_chunks