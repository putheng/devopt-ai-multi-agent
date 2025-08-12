from dotenv import load_dotenv
import os
from openai import OpenAI
from typing import List
import textwrap
from mem0 import MemoryClient

from qdrant_client import QdrantClient

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")

openai_client = OpenAI(api_key=openai_api_key)
qdrant_client = QdrantClient(host="localhost", port=6333)
memory_client = MemoryClient(api_key=os.getenv("MEM0_API_KEY"))

collection_name = "devopt_documents"
chat_model = "gpt-4o"
embedding_model="text-embedding-3-small"

def text_chunking(text, max_tokens=300) -> List[str]:
    # Approx 1 token = 4 characters in English
    chunk_size = max_tokens * 4
    return textwrap.wrap(text, chunk_size)

def text_embedding(text: str) -> List[float]:
    response = openai_client.embeddings.create(
        model=embedding_model,
        input=text,
    )
    return response.data[0].embedding

def vector_search(query_vector: List[float], top_k: int = 5) -> List[str]:

    search_result = qdrant_client.query_points(
        collection_name=collection_name,
        query=query_vector,
        limit=top_k,
        with_payload=True
    ).points

    matched_texts = [hit.payload["text"] for hit in search_result]
    return matched_texts

def retrieve(query: str, top_k: int = 5) -> List[str]:
    query_vector = text_embedding(query)
    return vector_search(query_vector, top_k)