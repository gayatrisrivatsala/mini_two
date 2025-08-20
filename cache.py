from typing import Dict, List
# Corrected Import: Removed the leading dot.
from schemas import Chunk


processed_document_cache: Dict[str, List[Chunk]] = {}