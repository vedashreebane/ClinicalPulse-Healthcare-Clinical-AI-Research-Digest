# defines the 4 team members — who they are, what they care about, and what tools they can use.

import os
from dotenv import load_dotenv
from crewai import Agent
from crewai.tools import tool
from langchain_openai import ChatOpenAI

from tools.news_tool import fetch_news
from tools.chroma_tool import store_chunks, retrieve_chunks
from tools.arxiv_tool import fetch_papers, chunk_papers

load_dotenv()

llm = "gpt-4o"  

# Wrap the tools in a way that they can be used by the agents.

@tool("Fetch recent news articles")
def news_fetcher_tool(query: str) -> str:
    """Fetches recent healthcare AI news articles for a given query."""
    return fetch_news(query)

@tool("Fetch and retrieve research papers")
def arxiv_and_retrieve_tool(query: str) -> str:
    """Fetches papers from ArXiv, stores them in ChromaDB, then retrieves the most relevant chunks. Always does both steps in order."""
    papers = fetch_papers(query)
    chunks = chunk_papers(papers)
    store_chunks(chunks)
    results = retrieve_chunks(query)
    return results

# Define the 4 agents with their roles, goals, backstories, and tools.
# Each agent has a specific focus and set of tools to help them achieve their goal.
def make_agents():

    news_retriever = Agent(
        role="Healthcare News Analyst",
        goal="Find the most recent and relevant healthcare AI news articles",
        backstory=(
            "You are an experienced medical journalist who tracks the latest "
            "developments in AI and healthcare. You know how to find and "
            "summarise the most impactful recent news stories."
        ),
        tools=[news_fetcher_tool],
        llm=llm,
        verbose=True
    )

    research_retriever = Agent(
        role="Clinical Research Specialist",
        goal="Find and retrieve the most relevant recent research papers from ArXiv",
        backstory=(
            "You are a clinical AI researcher with expertise in oncology and "
            "medical imaging. You specialise in finding cutting-edge academic "
            "research and extracting the key technical findings."
        ),
        tools=[arxiv_and_retrieve_tool],   # ← single combined tool
        llm=llm,
        verbose=True
    )

    synthesizer = Agent(
        role="Research Synthesizer",
        goal="Combine news and research findings into a coherent unified context",
        backstory=(
            "You are a senior editor who specialises in translating complex "
            "clinical AI research into clear, structured summaries. You are "
            "skilled at identifying connections between news trends and "
            "academic research."
        ),
        tools=[],
        llm=llm,
        verbose=True
    )

    report_writer = Agent(
        role="Medical Research Report Writer",
        goal="Write a clear, structured, cited report based on synthesised findings",
        backstory=(
            "You are a professional medical writer who produces high-quality "
            "research digests for clinicians and healthcare executives. Every "
            "claim you make is backed by a cited source."
        ),
        tools=[],
        llm=llm,
        verbose=True
    )

    return news_retriever, research_retriever, synthesizer, report_writer