from typing import List,Optional, Dict
import numpy as np

from model_loader import ModelAssets
from intent_classifier import get_emotion_filter

assets = ModelAssets()

model = assets.get_model()
index = assets.get_index()
metadata = assets.get_metadata()

def apply_mood_filter(results: List[Dict], mood_filter: Dict) -> List[Dict]:
    filtered = []

    for item in results:
        valence = item.get('valence')
        energy = item.get('energy')

        if (
            ("min_valence" in mood_filter and valence < mood_filter["min_valence"]) or
            ("max_valence" in mood_filter and valence > mood_filter["max_valence"]) or
            ("min_energy" in mood_filter and energy < mood_filter["min_energy"]) or
            ("max_energy" in mood_filter and energy > mood_filter["max_energy"])
        ):
            continue
        filtered.append(item)
    return filtered

def search_songs(query: str, top_k: int = 10) -> List[Dict]:

    # import pdb;pdb.set_trace()
    # get emotion based filter
    mood_filter = get_emotion_filter(query)

    # create embeddings for the query
    query_embedding = np.array(model.encode([query])).astype('float32')

    # search in the vector database
    _, indices = index.search(query_embedding, top_k * 2)

    # results = [metadata[int(i)] for i in indices[0] if i < len(metadata)]
    results = []
    for rank, idx in enumerate(indices[0]):
        row = metadata.iloc[idx]
        results.append(row)
    # results = apply_mood_filter(results, mood_filter)

    return results[:top_k]

songsArr = search_songs("song by asha bhosle")
for song in songsArr:
    print(song)
    print('==============================================')
