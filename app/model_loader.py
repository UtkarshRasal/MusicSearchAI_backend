import os
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer

class ModelAssets:
    def __init__(self, 
                index_path: str="faiss_index/mendeley_index.faiss", 
                metadata_path: str = "faiss_index/metadata.csv",
                model_name: str = "BAAI/bge-base-en"
    ):
        self.index_path = index_path    
        self.metadata_path = metadata_path    
        self.model_name = model_name    

        self.model = None
        self.index = None
        self.metadata = None

        self.load_all()

    def load_all(self):
        print("Loading Sentence Trasnformer Models...")
        self.model = SentenceTransformer(self.model_name)

        # Load FAISS index
        if not os.path.exists(self.index_path):
            raise FileNotFoundError(f"FAISS index not found at: {self.index_path}")
        
        print("📦 Loading FAISS index...")
        self.index = faiss.read_index(self.index_path)

        # Load metadata
        if not os.path.exists(self.metadata_path):
            raise FileNotFoundError(f"Metadata CSV not found at: {self.metadata_path}")
        
        print("📄 Loading metadata...")
        self.metadata = pd.read_csv(self.metadata_path)

        print("✅ All model components loaded successfully.")


    def get_model(self):
        return self.model
    
    def get_index(self):
        return self.index
    
    def get_metadata(self):
        return self.metadata