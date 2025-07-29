# /rag_project/retriever.py

from typing import List
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from schemas import Chunk

model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')

def find_relevant_chunks(question: str, all_chunks: List[Chunk], top_k: int = 5) -> List[Chunk]:
    question_embedding = model.encode([question], convert_to_numpy=True)
    chunk_embeddings = np.array([chunk.embedding for chunk in all_chunks])
    
    similarities = cosine_similarity(question_embedding, chunk_embeddings)[0]
    
    # Get the indices of the top_k chunks, ensuring they are sorted by relevance
    top_k_indices = np.argsort(similarities)[::-1][:top_k]
    
    return [all_chunks[i] for i in top_k_indices]