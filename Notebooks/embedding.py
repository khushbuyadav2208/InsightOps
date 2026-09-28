import pandas as pd
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

tickets = pd.read_csv("tickets.csv")
tickets["content"] = tickets["content"].fillna("").astype(str)

tickets["embedding"] = tickets["content"].apply(
    lambda x: model.encode(x).tolist()
)

tickets.to_csv("tickets_with_embeddings.csv", index=False)
