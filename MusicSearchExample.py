# !mkdir data

# !pip install transformers sentence-transformers faiss-cpu pandas

import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
df = pd.read_csv("/content/data/songs.csv")

# Combine title + lyrics for embedding
texts = (df['title'] + '-' + df['lyrics']).tolist()

# Create embeddings
embeddings = model.encode(texts, show_progress_bar=True)

# Save embeddings with FAISS
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings).astype("float32"))

# Save index and metadata
# create a folder "faiss" if some faiss error occurs
faiss.write_index(index, "faiss/song_index.index")
df.to_csv("faiss/song_metadata.csv", index=False)

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
index = faiss.read_index("faiss/song_index.index")
df = pd.read_csv("faiss/song_metadata.csv")

def search_similar_songs(query: str, top_k: int = 5):
    query_embedding = model.encode([query])
    query_embedding = np.array(query_embedding).astype("float32")

    D, I = index.search(query_embedding, top_k)

    print(f"\n🔍 Top {top_k} Results for: '{query}'\n")
    for rank, idx in enumerate(I[0]):
        row = df.iloc[idx]
        print(f"{rank+1}. {row['title']} – {row['artist']}")
        print(f"    Lyrics: {row['lyrics'][:100]}...\n")


search_similar_songs('song about cold sweat')
