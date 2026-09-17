from sentence_transformers import SentenceTransformer

model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

texts = [
    "The investor may lose capital.",
    "The client has a conservative risk profile."
]

vectors = model.encode(texts)

print(vectors.shape)