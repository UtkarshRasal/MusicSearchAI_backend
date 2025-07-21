import pandas as pd
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance, PointStruct
from tqdm import tqdm
from pathlib import Path
from extract_zip import extract_zipped_csv_file
import nltk, hashlib
import numpy as np

nltk.download('punkt')
nltk.download('punkt_tab')
from nltk.tokenize import sent_tokenize

# paths
DATA_PATH = Path('data/').resolve()
ZIPPED_FILE_NAME = "MusicDataset.zip"
CSV_FILE_NAME = "MusicDataset.csv"
EMBEDDINGS_FILE_NAME = "MusicEmbeddings.npz"
FULL_DATA_PATH = Path(f"{DATA_PATH}/{CSV_FILE_NAME}").resolve()
FULL_EMBEDDINGS_PATH = Path(f"{DATA_PATH}/{EMBEDDINGS_FILE_NAME}").resolve()
INDEX_PATH = Path("faiss_index/mendeley_index.faiss").resolve()
METADATA_PATH = Path("faiss_index/metadata.csv").resolve()

COLLECTION_NAME = "mendeley_songs"
CHUNK_SIZE = 3  # Number of sentences per chunk


if not FULL_DATA_PATH.exists():
    print(f"'{FULL_DATA_PATH}' doesn't exist, zip the file")
    extract_zipped_csv_file(FULL_DATA_PATH,ZIPPED_FILE_NAME)

client = QdrantClient(
    url="https://58cd0a15-6969-4eea-b657-d45cf1f9d923.europe-west3-0.gcp.cloud.qdrant.io",
    api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.4-3mGNJJnDoWZkyLzAHJzEP52a_nBsebRwS72jwNKpY",
)

# Drop existing collection (if needed)
if client.collection_exists(COLLECTION_NAME):
    client.delete_collection(COLLECTION_NAME)


# # load model
print("🔄 Loading embedding model...")
model = SentenceTransformer("BAAI/bge-base-en")

# Load data
print("📄 Loading dataset...")
df = pd.read_csv(FULL_DATA_PATH)
df = df.dropna(subset=["lyrics", "track_name", "artist_name"])

points = []
print("🧠 Preparing data for embeddings.")

texts_to_embed = []
metadata_list = []
for idx, row in tqdm(df.iterrows(), total=len(df)):
    lyrics = row["lyrics"]
    artist = row["artist_name"]
    track = row["track_name"]

    # Chunking lyrics
    sentences = sent_tokenize(lyrics)
    chunks = [" ".join(sentences[i:i+CHUNK_SIZE]) for i in range(0, len(sentences), CHUNK_SIZE)]

    for i, chunk in enumerate(chunks):
        doc_id = f"{row['track_name']}_{row['artist_name']}_{i}"
        text = f"{track} by {artist}. Lyrics: {chunk}"
        texts_to_embed.append(text)
        metadata_list.append({
            "id": doc_id,
            "track_name": row["track_name"],
            "artist_name": row["artist_name"],
            "chunk": chunk,
            "genre": row.get("genre", ""),
            "valence": row.get("valence", None),
            "energy": row.get("energy", None),
        })

        # embedding = model.encode(text)

        # points.append(
        #     PointStruct(
        #         id=str(uuid.uuid4()),
        #         vector=embedding.tolist(),
        #         payload={
        #             "artist": artist,
        #             "track": track,
        #             "chunk": chunk
        #         }
        #     )
        # )


if not FULL_EMBEDDINGS_PATH.exists():
    print("⚙️ Generating new embeddings...")
    embeddings = model.encode(texts_to_embed, batch_size=32, show_progress_bar=True)

    # Save to cache
    print("💾 Saving to cache...")
    np.savez(FULL_EMBEDDINGS_PATH, embeddings=embeddings, metadata=metadata_list)
else:
    print("⚙️ Fetching existing embeddings...")
    data = np.load(FULL_EMBEDDINGS_PATH, allow_pickle=True)
    embeddings = data["embeddings"]
    metadata_list = data["metadata"].tolist()


# --- Step 3: Upload to Qdrant ---
print("📤 Uploading to Qdrant...")

# Create new collection
client.create_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=VectorParams(size=embeddings.shape[1], distance=Distance.COSINE),
)

points = [
    PointStruct(
        id=int(hashlib.md5(m["id"].encode()).hexdigest(), 16) % (10**9),
        vector=vector.tolist(),
        payload=m
    )
    for vector, m in zip(embeddings, metadata_list)
]

client.upload_points(collection_name=COLLECTION_NAME, points=points, batch_size=256)

print(f"✅ Uploaded {len(points)} vectors to Qdrant collection: {COLLECTION_NAME}")
print("✅ Build complete.")
