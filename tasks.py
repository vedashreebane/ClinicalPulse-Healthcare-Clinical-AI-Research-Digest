# defines the 4 jobs — the specific instructions for each agent on this particular query.

from crewai import Task
from agents import make_agents

def make_tasks(query: str):

    news_retriever, research_retriever, synthesizer, report_writer = make_agents()

    # Task 1: Fetch news 
    news_task = Task(
        description=(
            f"Search for recent news articles about: {query}\n"
            "Fetch the top 10 most recent articles and return them "
            "with their titles, sources, publication dates, URLs, and snippets."
        ),
        expected_output=(
            "A structured list of 10 recent news articles with title, "
            "source, date, URL and snippet for each."
        ),
        agent=news_retriever
    )

    # Task 2: Fetch + retrieve research
    research_task = Task(
        description=(
            f"Search ArXiv for recent research papers about: {query}\n"
            "First fetch and store the papers using the arxiv tool. "
            "Then retrieve the most relevant chunks using the retriever tool. "
            "Return the most relevant excerpts with full citation details."
        ),
        expected_output=(
            "A structured list of the most relevant research excerpts, "
            "each with paper title, authors, publication date, URL and excerpt."
        ),
        agent=research_retriever
    )

    # Task 3: Synthesize both sources 
    synthesis_task = Task(
        description=(
            f"You have been given news articles and research findings about: {query}\n"
            "Combine both sources into a unified context. "
            "Identify the key themes, trends, and findings that appear "
            "across both the news and the research. "
            "Note any connections or contradictions between the two sources."
        ),
        expected_output=(
            "A coherent synthesis of the news and research findings, "
            "organised by key themes with connections clearly identified."
        ),
        agent=synthesizer,
        context=[news_task, research_task]   # gets output from both previous tasks
    )

    # Task 4: Write the report 
    report_task = Task(
        description=(
           f"Write a professional research digest report on: {query}\n"
            "Use the synthesised findings to write a structured report with these sections:\n"
            "1. Executive Summary (3-4 sentences)\n"
            "2. Key Findings from Research (bullet points with citations)\n"
            "3. Recent News & Industry Trends (bullet points with sources)\n"
            "4. Conclusion & Implications\n"
            "5. References (all full URLs used — never write 'URL' as a placeholder, always use the actual full URL)\n\n"
            "Every claim must be backed by a citation.\n"
            "Use this exact format for citations: [Source: title, https://actual-url.com]\n"
            "In the References section, always write out the complete URL in full."
        ),
        expected_output=(
                "A fully structured research digest report with 5 sections, "
                "written in professional language with full URL citations for every claim. "
                "No placeholder text like 'URL' — only real links."
        ),
        agent=report_writer,
        context=[synthesis_task]   # gets the synthesized context
    )

    return [news_task, research_task, synthesis_task, report_task]