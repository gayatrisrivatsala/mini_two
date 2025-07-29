# /rag_project/llm_handler.py

import httpx
from typing import List
import logging
from config import settings
from schemas import Chunk

logger = logging.getLogger(__name__)

async def ask_mistral(question: str, relevant_chunks: List[Chunk]) -> str:
    context_parts = [f"CONTEXT from Page {chunk.page_number}:\n{chunk.text}" for chunk in relevant_chunks]
    context = "\n\n---\n\n".join(context_parts)
    
    prompt = f"""
You are an AI assistant for insurance policy analysis. Your task is to answer the user's question based *exclusively* on the provided context.

**Strict Instructions:**
1.  Your answer must be derived *only* from the text in the "CONTEXT" section.
2.  Do not use any external knowledge or make assumptions.
3.  When you find the answer, you must make the answer precise and to the point.
4.  If the context does not contain the answer, you MUST respond with exactly this phrase: "This information could not be found in the provided document."
5.  the response should be atmost 100 words max and also should be clean from special characters and should be in a proper format.
6.  If there is a chance for the answer to be a number, you must make the answer a number.but dont makeup the number.only provide one when there is one.
---
**CONTEXT:**
{context}
---

**QUESTION:** {question}

**ANSWER:**
"""

    headers = {"Authorization": f"Bearer {settings.MISTRAL_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": "mistral-small-latest",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0, # Set to 0.0 for maximum factuality
        "max_tokens": 600
    }
    
    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            response = await client.post("https://api.mistral.ai/v1/chat/completions", headers=headers, json=payload)
            response.raise_for_status()
            result = response.json()
            return result["choices"][0]["message"]["content"].strip()
        except httpx.HTTPStatusError as e:
            logger.error(f"Mistral API error: {e.response.status_code} - {e.response.text}")
            return f"Error: LLM API returned status {e.response.status_code}."
        except Exception as e:
            logger.error(f"Unexpected error contacting the LLM: {e}")
            return "Error: Could not get a response from the language model."