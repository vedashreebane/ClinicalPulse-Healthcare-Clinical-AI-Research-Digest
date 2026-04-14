# What this file does 
# It connects to NewsAPI, searches for recent articles on whatever query the user gives, 
# and returns them in a clean format that our agents can read later.

# Three simple steps inside the file:
# Connect to NewsAPI using your key
# Search for articles
# Return them as clean, readable text

from newsapi import NewsApiClient
import os
from dotenv import load_dotenv

load_dotenv()

def fetch_news(query: str) -> str:
    # Fetches recent news articles for a given query.
    # Returns them as a formatted string for the agent to read.
    client = NewsApiClient(api_key=os.getenv("NEWS_API_KEY"))

    response = client.get_everything(
        q=query,
        language="en",
        sort_by="publishedAt",    # most recent first
        page_size=10,
    )

    articles = response.get("articles", [])

    if not articles:
        return "No articles found."

    result = ""
    for i, article in enumerate(articles, 1):
        result += f"""
[Article {i}]
Title:     {article.get('title')}
Source:    {article.get('source', {}).get('name')}
Published: {article.get('publishedAt')}
URL:       {article.get('url')}
Snippet:   {article.get('content', '')[:300]}
---
"""
    return result

if __name__ == "__main__":
    output = fetch_news("AI cancer detection")
    print(output)