# What this file does
# It connects to the ArXiv API, searches for recent research papers, and chunks them up ready for embedding.
# Two functions inside:
# fetch_papers() — searches ArXiv and returns paper metadata + abstracts
# chunk_papers() — splits each abstract into overlapping chunks

# tools/arxiv_tool.py

import arxiv

CHUNK_SIZE = 500      # characters per chunk
CHUNK_OVERLAP = 50    # characters repeated between chunks


def fetch_papers(query: str) -> list[dict]:
    # Fetches top 5 recent papers from ArXiv for a given query.

    client = arxiv.Client()

    # Add medical/biology category filter to keep results relevant
    filtered_query = f"{query} AND (cat:cs.AI OR cat:cs.LG OR cat:eess.IV OR cat:q-bio.QM)"

    search = arxiv.Search(
        query=query,
        max_results=5,
        sort_by=arxiv.SortCriterion.Relevance,
        sort_order=arxiv.SortOrder.Descending,
    )

    papers = []
    for result in client.results(search):
        papers.append({
            "title":     result.title,
            "abstract":  result.summary,
            "authors":   ", ".join(str(a) for a in result.authors[:5]),
            "published": str(result.published.date()),
            "url":       result.entry_id,
        })

    print(f"Fetched {len(papers)} papers")
    return papers


def chunk_text(text: str) -> list[str]:
    # Splits text into overlapping chunks of CHUNK_SIZE characters.
    chunks = []
    start = 0

    while start < len(text):
        end = start + CHUNK_SIZE
        chunks.append(text[start:end])
        start += CHUNK_SIZE - CHUNK_OVERLAP

    return chunks


def chunk_papers(papers: list[dict]) -> list[dict]:
    # Takes papers and splits each one into chunks.
    # Each chunk keeps the paper's metadata attached for citations later.
    all_chunks = []

    for i, paper in enumerate(papers):
        full_text = f"{paper['title']}\n\n{paper['abstract']}"
        chunks = chunk_text(full_text)

        for j, chunk in enumerate(chunks):
            all_chunks.append({
                "chunk_id":  f"paper_{i}_chunk_{j}",
                "text":      chunk,
                "title":     paper["title"],
                "authors":   paper["authors"],
                "published": paper["published"],
                "url":       paper["url"],
            })

    print(f"Split into {len(all_chunks)} chunks")
    return all_chunks

if __name__ == "__main__":
    papers = fetch_papers("AI cancer detection")
    chunks = chunk_papers(papers)

    print(f"\nSample chunk:\n{chunks[0]['text']}")
    print(f"\nFrom: {chunks[0]['title']}")
    print(f"URL:  {chunks[0]['url']}")