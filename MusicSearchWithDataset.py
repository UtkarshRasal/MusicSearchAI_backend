# This code is copied from a jupyter notebook, might have some errors or might need to re structure the code properly

# !mkdir data

# !mkdir faiss

# !pip install transformers sentence-transformers faiss-cpu pandas
# !pip install faiss-cpu pandas

import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('BAAI/bge-base-en')
df = pd.read_csv("/content/data/tcc_ceds_music.csv")

# drop missing data
df = df.dropna(subset=['lyrics', 'track_name', 'artist_name'])

# Fill valence with 0.5 if missing (neutral)
if 'valence' not in df.columns:
    df['valence'] = 0.5
else:
    df['valence'] = df['valence'].fillna(0.5)

# Combine fields into a searchable text
def make_text(row):
    return (
        f"{row['track_name']} by {row['artist_name']}. "
        f"Genre: {row['genre']}. "
        f"Valence: {row['valence']:.2f}. "
        f"Lyrics: {row['lyrics']}"
    )

df['combined'] = df.apply(make_text, axis=1)
df

# Create embeddings
embeddings = model.encode(df['combined'].tolist(), normalize_embeddings=True)
embeddings = np.array(embeddings).astype('float32')

# Save embeddings with FAISS
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)


import os

# Save index and metadata
# create a folder "faiss" if some faiss error occurs
os.makedirs("faiss_index", exist_ok=True)
faiss.write_index(index, "faiss_index/mendeley_index.faiss")

# Save metadata
df[['track_name', 'artist_name', 'genre', 'valence', 'lyrics']].to_csv("faiss_index/metadata.csv", index=False)

def load_resources():
    index = faiss.read_index("faiss_index/mendeley_index.faiss")
    df = pd.read_csv("faiss_index/metadata.csv")
    return index, df

def search(query, top_k=5):
    query_vec = model.encode([query], normalize_embeddings=True).astype("float32")
    D, I = index.search(query_vec, top_k)
    # return df.iloc[I[0]]

    print(f"\n🔍 Top {top_k} Results for: '{query}'\n")
    for rank, idx in enumerate(I[0]):
        row = df.iloc[idx]
        print(f"{rank+1}. {row['track_name']} – {row['artist_name']}")
        print(f"    Lyrics: {row['lyrics'][:100]}...\n")

# Load
index, df = load_resources()

# Try example searches
print(search("sad romantic piano song"))
print('=========================================================================')
print(search("happy energetic dance song"))
print('=========================================================================')
print(search("lonely heartbreak lyrics"))
print('=========================================================================')
print(search("song by frankie laine"))
print('=========================================================================')
print(search("most emotional song"))

