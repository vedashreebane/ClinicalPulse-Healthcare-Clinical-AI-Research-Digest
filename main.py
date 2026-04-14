from crewai import Crew, Process
from agents import make_agents
from tasks import make_tasks


def run(query: str):
    print(f"\n{'='*60}")
    print(f"  ClinicalPulse — Research Digest")
    print(f"  Query: {query}")
    print(f"{'='*60}\n")

    agents = make_agents()
    tasks  = make_tasks(query)

    crew = Crew(
        agents=list(agents),
        tasks=tasks,
        process=Process.sequential,   # agents run one after another
        verbose=True
    )

    result = crew.kickoff()

    print(f"\n{'='*60}")
    print("  FINAL REPORT")
    print(f"{'='*60}\n")
    print(result)

    # save report to a file
    with open("report.md", "w") as f:
        f.write(f"# ClinicalPulse Report\n**Query:** {query}\n\n")
        f.write(str(result))
    print("\nReport saved to report.md")


if __name__ == "__main__":
    query = input("Enter your research query: ")
    run(query)