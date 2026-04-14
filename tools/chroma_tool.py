# What chroma_tool.py does
# This is the heart of RAG. It has two jobs:
# Store — take the chunks from arxiv_tool.py, convert them to vectors using OpenAI, and save them in ChromaDB locally on your machine
# Retrieve — when a query comes in, convert it to a vector and find the most similar chunks from ChromaDB

import os
import chromadb
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

CHROMA_DIR        = "./chroma_db"        # saved locally on your machine
COLLECTION_NAME   = "clinicalpulse"
EMBEDDING_MODEL   = "text-embedding-3-small"
TOP_K             = 5                    # how many chunks to retrieve


def get_collection():
    # Connects to (or creates) a local ChromaDB collection.

    client = chromadb.PersistentClient(path=CHROMA_DIR)
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}   # use cosine similarity
    )


def embed(texts: list[str]) -> list[list[float]]:
    # Converts a list of text strings into vectors using OpenAI.
    # Returns one vector per input string.

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=texts
    )
    return [item.embedding for item in response.data]

def store_chunks(chunks: list[dict]) -> None:
    # Embeds chunks and stores them in ChromaDB.
    # Skips chunks that are already stored (avoids duplicate embeddings).

    collection = get_collection()

    # filter out chunks we already stored
    existing   = collection.get(ids=[c["chunk_id"] for c in chunks])
    existing_ids = set(existing["ids"])
    new_chunks = [c for c in chunks if c["chunk_id"] not in existing_ids]

    if not new_chunks:
        print("All chunks already stored — skipping embedding.")
        return

    print(f"Embedding {len(new_chunks)} new chunks...")

    texts      = [c["text"] for c in new_chunks]
    embeddings = embed(texts)

    collection.add(
        ids        = [c["chunk_id"] for c in new_chunks],
        embeddings = embeddings,
        documents  = texts,
        metadatas  = [
            {
                "title":     c["title"],
                "authors":   c["authors"],
                "published": c["published"],
                "url":       c["url"],
            }
            for c in new_chunks
        ]
    )
    print(f"Stored {len(new_chunks)} chunks. Total in DB: {collection.count()}")


def retrieve_chunks(query: str) -> str:
    # Finds the most relevant chunks for a query.
    # Returns them as a formatted string ready for the agent.
    collection = get_collection()

    if collection.count() == 0:
        return "No research found. Run store_chunks() first."

    # embed the query using the same model
    query_vector = embed([query])[0]

    results = collection.query(
        query_embeddings = [query_vector],
        n_results        = min(TOP_K, collection.count()),
        include          = ["documents", "metadatas", "distances"]
    )

    output = ""
    for i in range(len(results["ids"][0])):
        score = 1 - results["distances"][0][i]   # cosine: 1=identical, 0=unrelated
        meta  = results["metadatas"][0][i]
        text  = results["documents"][0][i]

        output += f"""
[Research Chunk {i+1}] — relevance: {score:.2f}
Paper:     {meta['title']}
Authors:   {meta['authors']}
Published: {meta['published']}
URL:       {meta['url']}
Excerpt:   {text}
---
"""
    return output

if __name__ == "__main__":
    from arxiv_tool import fetch_papers, chunk_papers

    print("Fetching papers...")
    papers = fetch_papers("AI cancer detection")
    chunks = chunk_papers(papers)

    print("\nStoring chunks in ChromaDB...")
    store_chunks(chunks)

    print("\nRetrieving relevant chunks...")
    results = retrieve_chunks("how does deep learning detect tumors?")
    print(results)