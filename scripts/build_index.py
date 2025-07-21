import pandas as pd
import numpy as np
import faiss
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from pathlib import Path
from extract_zip import extract_zipped_csv_file

# paths
DATA_PATH = Path('data/').resolve()
ZIPPED_FILE_NAME = "MusicDataset.zip"
CSV_FILE_NAME = "MusicDataset.csv"
FULL_DATA_PATH = Path(f"{DATA_PATH}/{CSV_FILE_NAME}").resolve()
INDEX_PATH = Path("faiss_index/mendeley_index.faiss").resolve()
METADATA_PATH = Path("faiss_index/metadata.csv").resolve()

if not FULL_DATA_PATH.exists():
    print(f"'{FULL_DATA_PATH}' doesn't exist, zip the file")
    extract_zipped_csv_file(FULL_DATA_PATH,ZIPPED_FILE_NAME)

# # load model
print("🔄 Loading embedding model...")
model = SentenceTransformer("BAAI/bge-base-en")

# load dataset
print("📄 Loading dataset...")
df = pd.read_csv(FULL_DATA_PATH)

# fillna
df.fillna("", inplace=True)

# Fill valence with 0.5 if missing (neutral)
if 'valence' not in df.columns:
    df['valence'] = 0.5
else:
    df['valence'] = df['valence'].fillna(0.5)

# Columns to keep as metadata
metadata_cols = ["track_name", "artist_name", "genre", "valence", "energy", "lyrics"]

# construct searchable text
print('🧠 Building metdata..')
metadata = []
for _, row in tqdm(df.iterrows(), total=len(df)):
    text = (
        f"{row['track_name']} by {row['artist_name']}. "
        f"Genre: {row['genre']}. "
        f"Lyrics: {row['lyrics']}. "
        f"Valence: {row['valence']:.2f}, Energy: {row['energy']:.2f}."
    )
    metadata.append(text.strip())

# Generate embeddings
print("⚙️ Generating embeddings...")
embeddings = model.encode(
    metadata,
    batch_size=32,
    show_progress_bar=True,
    normalize_embeddings=True
)

embeddings = np.array(embeddings).astype('float32')

# Build FAISS index
print("📦 Creating FAISS index...")
dim = embeddings.shape[1]
index = faiss.IndexFlatIP(dim)  # Cosine similarity
index.add(embeddings)

# Save index
print(f"Writing to index_path - {INDEX_PATH}")
faiss.write_index(index, str(INDEX_PATH))
print(f"✅ Saved FAISS index to {INDEX_PATH}")

# Save metadata
df[metadata_cols].to_csv(str(METADATA_PATH), index=False)
print(f"✅ Saved metadata to {METADATA_PATH}")

