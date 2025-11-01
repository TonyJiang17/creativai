from sentence_transformers import SentenceTransformer
import re

def surprise_score(input):
    model = SentenceTransformer("thenlper/gte-large", use_auth_token=True)
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', input) if s.strip()]
    if len(sentences) < 2:
        raise Exception("Input is too short")
    input_embedding = model.encode([input])
    sentence_embeddings = model.encode(sentences)
    similarities = model.similarity(input_embedding, sentence_embeddings)[0]
    distances = [1 - s for s in similarities]
    total = 0
    for i in range(1, len(distances)):
        total += distances[i] - distances[i - 1]
    breakpoint()
    score = 2 * total / (len(distances) - 1)
    return score

if __name__ == "__main__":
    with open("texts/sample_aitah/s1.txt", "r", encoding="utf-8") as f:
        text = f.read()
        print(surprise_score(text))