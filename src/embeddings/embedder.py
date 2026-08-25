from sentence_transformers import SentenceTransformer
from huggingface_hub import snapshot_download
import torch

class Embedder:
    def __init__(self, model_name="BAAI/bge-m3", device=None):
        # Fix: Check the incoming argument and assign directly to self.device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        model_path = snapshot_download(repo_id=model_name)
        # Fix: Pass self.device instead of the local variable
        self.model = SentenceTransformer(
            model_path, 
            device=self.device
        )
        
    def embed_documents(self, texts):
        return self.model.encode(
            texts, 
            batch_size=8, 
            convert_to_numpy=True, 
            normalize_embeddings=True, 
            show_progress_bar=True
        )
        
    def embed_query(self, query):
        return self.model.encode(
            query, 
            convert_to_numpy=True, 
            normalize_embeddings=True
        )
