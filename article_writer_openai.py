# research and write an article

# warning control
import warnings
warnings.filterwarnings('ignore')

from crewai import Agent, Task, Crew
import os
from utils import get_openai_api_key

# Set API key
openai_api_key = get_openai_api_key()
os.environ["OPENAI_API_KEY"] = openai_api_key
os.environ['OPENAI_MODEL_NAME'] = 'gpt-3.5-turbo'


# --------------------
# Agent 1: Planner
# --------------------
planner = Agent(
    role='Content Planner',
    goal='Plan engaging and factually correct content on {topic}',
    backstory=(
        'You are a researcher. Research about the topic {topic} and think '
        'how to introduce the idea to the general public. Also think about '
        'how to elaborate on the idea, which details to include, and plan '
        'a structured way to present the topic.'
    ),
    allow_delegation=False,
    verbose=True
)

# --------------------
# Agent 2: Writer
# --------------------
writer = Agent(
    role='Content Writer',
    goal='Write engaging and factually correct content on {topic}',
    backstory=(
        'You are a content writer. You follow the presentation plan given '
        'by the content planner and write a well-structured article.'
    ),
    allow_delegation=False,
    verbose=True
)

# --------------------
# Agent 3: Editor
# --------------------
editor = Agent(
    role='Content Editor',
    goal='Edit the content on {topic}',
    backstory=(
        'You edit the content written by the writer. '
        'Check for grammatical errors, clarity, coherence, and vocabulary.'
    ),
    allow_delegation=False,
    verbose=True
)

# --------------------
# Tasks
# --------------------

plan_task = Task(
    description='Plan engaging and factually correct content on {topic}.',
    expected_output='A structured content plan with headings and subtopics.',
    agent=planner
)

write_task = Task(
    description='Write an article on the topic {topic} as planned by the planner.',
    expected_output=(
        'A complete article on the topic {topic}. '
        'There should be 3 to 4 paragraphs under each subtopic. '
        'Each paragraph should contain a maximum of 15 lines.'
    ),
    agent=writer
)

edit_task = Task(
    description='Edit the content written by the writer.',
    expected_output=(
        'A grammatically correct article with improved clarity and vocabulary.'
    ),
    agent=editor
)

# --------------------
# Crew
# --------------------

crew = Crew(
    agents=[planner, writer, editor],
    tasks=[plan_task, write_task, edit_task],
    verbose=True
)

# Run the crew
result = crew.kickoff(inputs={"topic": "Artificial Intelligence"})

print(result)