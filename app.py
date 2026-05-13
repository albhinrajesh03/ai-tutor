from pdf_loader import load_pdf
from basic_rag import split_text, retrieve
from llm import ask_llm

text = load_pdf("book.pdf")
chunks = split_text(text)

print("AI Tutor (type 'bye' to exit)")

while True:
    question = input("You: ")

    if question.lower() == "bye":
        print("Goodbye")
        break

    result = []
    result = retrieve(question, chunks)

    if not result:
        print("Ai: I couldn't find relevant information in the document.")
        continue

    context="\n".join(result)

    prompt = f"""
    You are a strict AI tutor.

    Your job:
        - Teach only using the given context
        - If context is insufficient, say: "Not found in the document"
        - Never hallucinate or assume information

    Response style:
        - Simple explanation
        - Step-by-step format
        - Use examples if possible
        - End with 1 question to test understanding

    Context: {context}
    User Question:{question}
    """

    answer = ask_llm(prompt)

    print("AI:")
    print(answer)
    print("\n" + "-"*50 + "\n")