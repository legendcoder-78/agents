import os
from crewai import Agent, Task, Crew, LLM

# 1) Set your Gemini API key in terminal first:
# export GOOGLE_API_KEY="your_key_here"
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise RuntimeError("Set GOOGLE_API_KEY in your terminal and run again.")

# 2) Create one LLM
llm = LLM(
    model="gemini/gemini-2.5-flash",
    api_key=api_key,
)

# 3) Create three agents: planner, writer, editor
planner = Agent(
    role="Planner",
    goal="Create a simple article outline on {topic}",
    backstory="You organize ideas clearly before writing starts.",
    llm=llm,
    verbose=True,
)

writer = Agent(
    role="Writer",
    goal="Write a clear article on {topic}",
    backstory="You turn outlines into readable content.",
    llm=llm,
    verbose=True,
)

editor = Agent(
    role="Editor",
    goal="Improve grammar and clarity of the article on {topic}",
    backstory="You polish writing to make it easy to read.",
    llm=llm,
    verbose=True,
)

# 4) Create tasks for each agent
plan_task = Task(
    description="Create a short outline for an article on {topic}.",
    expected_output="A simple bullet-point outline.",
    agent=planner,
)

write_task = Task(
    description="Write a 2-paragraph article on {topic} using the outline.",
    expected_output="A draft article with 2 paragraphs.",
    agent=writer,
)

edit_task = Task(
    description="Edit the draft for grammar, flow, and clarity.",
    expected_output="A polished 2-paragraph article.",
    agent=editor,
)

# 5) Run crew
crew = Crew(
    agents=[planner, writer, editor],
    tasks=[plan_task, write_task, edit_task],
    verbose=True,
)
result = crew.kickoff(inputs={"topic": "Artificial Intelligence"})

print("\n=== FINAL ARTICLE ===\n")
print(result)
