import os
import warnings

# Optional: keep local runs quiet and avoid telemetry network calls.
os.environ.setdefault("CREWAI_DISABLE_TELEMETRY", "true")

try:
    from crewai import Agent, Task, Crew, LLM
    from crewai.memory.storage.kickoff_task_outputs_storage import (
        KickoffTaskOutputsSQLiteStorage,
    )
except ModuleNotFoundError as e:
    raise SystemExit(
        "Missing dependency: crewai. Activate your virtualenv (e.g. ./crewenv/bin/python) "
        "or install with: pip install crewai"
    ) from e

# 1. SETUP - API key
# Preferred: export GOOGLE_API_KEY or GEMINI_API_KEY in your shell.
# Optional fallback: paste your key into DIRECT_API_KEY below.
DIRECT_API_KEY = ""

MY_API_KEY = (
    os.getenv("GOOGLE_API_KEY")
    or os.getenv("GEMINI_API_KEY")
    or DIRECT_API_KEY
)
if not MY_API_KEY:
    raise RuntimeError(
        "Missing API key. Set GOOGLE_API_KEY (or GEMINI_API_KEY), "
        "or paste it into DIRECT_API_KEY."
    )

# Set BOTH possible names to satisfy every internal library
os.environ["GEMINI_API_KEY"] = MY_API_KEY
os.environ["GOOGLE_API_KEY"] = MY_API_KEY

warnings.filterwarnings('ignore')

# 2. INITIALIZE LLM 
# 'gemini/' prefix triggers AI Studio logic instead of Vertex AI
MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini/gemini-2.5-flash")
llm = LLM(
    model=MODEL_NAME,
    api_key=MY_API_KEY,      # Direct injection
    temperature=0.5,
    max_tokens=2000          # Added to ensure request is well-formed
)

# 3. AGENTS
planner = Agent(
    role='Content planner',
    goal='Plan engaging content on {topic}',
    backstory='Researcher who creates structured outlines.',
    llm=llm,
    verbose=True,
    allow_delegation=False
)

writer = Agent(
    role='Content writer',
    goal='Write a 2-paragraph article on {topic}',
    backstory='You transform outlines into engaging prose.',
    llm=llm,
    verbose=True,
    allow_delegation=False
)

editor = Agent(
    role='Content editor',
    goal='Refine vocabulary and grammar on {topic}',
    backstory='Meticulous editor focused on clarity.',
    llm=llm,
    verbose=True,
    allow_delegation=False
)

# 4. TASKS
plan = Task(
    description='Create a structured outline for {topic}.',
    expected_output='A bullet-point outline.',
    agent=planner
)

write = Task(
    description="Write a 2-paragraph article based on the outline.",
    expected_output="Two short paragraphs.",
    agent=writer
)

edit = Task(
    description='Fix grammar and flow.',
    expected_output='A polished 2-paragraph article.',
    agent=editor
)

# 5. CREW
crew = Crew(
    agents=[planner, writer, editor],
    tasks=[plan, write, edit],
    verbose=True
)

# Ensure CrewAI writes its sqlite file into this project (writable here).
local_db_dir = os.path.join(os.getcwd(), ".crewai_storage")
os.makedirs(local_db_dir, exist_ok=True)
crew._task_output_handler.storage = KickoffTaskOutputsSQLiteStorage(
    db_path=os.path.join(local_db_dir, "latest_kickoff_task_outputs.db")
)

# Execution
result = crew.kickoff(inputs={"topic": "Artificial Intelligence"})

print("\n" + "="*30)
print("FINAL POLISHED ARTICLE")
print("="*30)
print(result)
