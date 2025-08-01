# /llm_handler.py

import httpx
from typing import List
import logging
import asyncio

from config import settings
from schemas import Chunk

logger = logging.getLogger(__name__)

async def _call_mistral_api(payload: dict, semaphore: asyncio.Semaphore) -> dict:
    """A single, rate-limited function to call the Mistral API."""
    headers = {"Authorization": f"Bearer {settings.MISTRAL_API_KEY}", "Content-Type": "application/json"}
    
    async with semaphore: # Wait for an available concurrency slot
        async with httpx.AsyncClient(timeout=60.0) as client:
            try:
                # Cooldown to ensure we don't hit the "requests per minute" limit.
                await asyncio.sleep(1) 
                
                response = await client.post("https://api.mistral.ai/v1/chat/completions", headers=headers, json=payload)
                response.raise_for_status()
                return response.json()
            except httpx.HTTPStatusError as e:
                logger.error(f"Mistral API error: {e.response.status_code} - {e.response.text}")
                return {"error": f"LLM API returned status {e.response.status_code}."}
            except Exception as e:
                logger.error(f"Unexpected error contacting the LLM: {e}")
                return {"error": "Could not get a response from the language model."}

def _clean_text(raw_text: str) -> str:
    """A robust function to clean final model output."""
    # --- The Definitive Cleaning Fix ---
    # Replace backslashes, newlines, and tabs
    text = raw_text.replace('\\', '').replace('\n', ' ').replace('\t', ' ').replace('/', '').replace(';', '')
    
    # Replace multiple spaces with a single space and strip leading/trailing whitespace
    return ' '.join(text.split()).strip()


async def generate_hypothetical_answer(question: str, semaphore: asyncio.Semaphore) -> str:
    """Generates a hypothetical answer."""
    prompt = f"As an expert insurance analyst, write a concise, hypothetical answer to the following question, as if it were extracted from a policy document.\n\nQUESTION: {question}\n\nHYPOTHETICAL ANSWER:"
    payload = {"model": "mistral-small-latest", "messages": [{"role": "user", "content": prompt}], "temperature": 0.5, "max_tokens": 200}
    
    result = await _call_mistral_api(payload, semaphore)
    if "error" in result:
        return question # Fallback to original question on error
    
    # Clean the hypothetical answer as well
    return _clean_text(result["choices"][0]["message"]["content"])


async def ask_mistral(question: str, relevant_chunks: List[Chunk], semaphore: asyncio.Semaphore) -> str:
    """Generates a final answer using the POWERFUL model and cleans the output."""
    context_parts = [f"CONTEXT from Page {chunk.page_number}:\n{chunk.text}" for chunk in relevant_chunks]
    context = "\n\n---\n\n".join(context_parts)
    prompt = f"""You are an AI assistant for insurance policy analysis. Your task is to answer the user's question based *exclusively* on the provided context.

**Strict Instructions:**
1.  Your answer must be derived *only* from the text in the "CONTEXT" section.
2.  Do not use any external knowledge or assumptions.
3.  Be precise and to the point.
4.  If the context does not contain the answer, you MUST respond with "This information could not be found in the provided document."
5.  If the answer involves a numerical value, state it clearly. Do not make up numbers.

---
**CONTEXT:**
{context}
---

**QUESTION:** {question}

**ANSWER:**
"""
    payload = {"model": "mistral-large-latest", "messages": [{"role": "user", "content": prompt}], "temperature": 0.0, "max_tokens": 600}
    
    result = await _call_mistral_api(payload, semaphore)
    
    if "error" in result:
        return result["error"]
        
    raw_answer = result["choices"][0]["message"]["content"]
    
    # Use the enhanced cleaning function for the final output
    return _clean_text(raw_answer)