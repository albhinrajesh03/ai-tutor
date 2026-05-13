import requests

def ask_llm(prompt):
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "phi3:mini",
                "prompt": prompt,
                "stream": False
            },
        )
        return response.json()["response"]
    except Exception as e:
        return f"Error: {e}"