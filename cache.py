from typing import Dict, List
# Corrected Import: Removed the leading dot.
from schemas import Chunk

# Simple in-memory dictionary to cache processed documents.
# The key is the document URL and the value is the list of Chunk objects.
processed_document_cache: Dict[str, List[Chunk]] = {}