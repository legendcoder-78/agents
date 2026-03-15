import warnings
import os
warnings.filterwarnings('ignore')
from crewai import Agent,Task,Crew,LLM
from utils import serper_api_key,GEMINI_API_KEY
from crewai_tools import ScrapeWebsiteTool,SerperDevTool
serper_api_key=serper_api_key()
api_key=GEMINI_API_KEY()
llm=LLM(
    model='gemini/gemini-2.5-flash',
    api_key=api_key
)
search_tool=SerperDevTool()
scrape_tool=ScrapeWebsiteTool()
#Agent 1-Venue Coordinator
ven_coor=Agent(
    role='Venue coordinator',
    goal='To search and find the best suited venue for an event considering factors like event type, budget and vibe',
    backstory='You are a venue coordinataor for a event {event}. Your job is to search for appropriate venue locations for the event keeping the type of event, budget constraints and size of group attending the event',
    tools=[scrape_tool,search_tool],
    allow_delegation=False,
    llm=llm
    )
#Agent 2
log_man=Agent(
    role='Logistics Manager',
    goal='to prepare a list of everything needed for the event',
    backstory='You are the logistics manager for the event {event}.You prepare a list of logistics needed at the required time and place.You also create a timeline for the setup operations .',
    tools=[search_tool,scrape_tool],
    verbose=True,
        allow_delegation=False,
        llm=llm

)
#Agent 3
marketing_agent=Agent(
    role='Marketing and Communications Agent',
    goal='To create content for effectively marketing the event ',
    backstory='You are a creative and collaborative Marketing and Communications Agent.You create compelling content which ensure maximum participation',
    allow_delegation=False,
    verbose=True,
    llm=llm
)
# creating a pydantic object to structrually store the results a tasks 
from pydantic import BaseModel
class VenueDetails(BaseModel):
    name:str
    address:str
    capacity:str
    booking_status:str
#Tasks
ven_coordinator=Task(
    description='find the best suited venue for the event {event} in {country},keeping budget of {budget} in rupees in mind.',
    expected_output='Details of the venue chosen by you',
    agent=ven_coor,
    human_input=True,
    output_json=VenueDetails,
    output_file='Venue_details.json',
)
log_task=Task(
    description='You Plan logistical requirements for the event {event} keeping the probable number of attendies as 500.',
    expected_output='Detailed list of logistical requirements along with timeline',
    agent=log_man,
    human_input=True,
    async_execution=True,
)
mark_task=Task(
    description='You have to creatively market the event {event} to ensure maximum participation',
    expected_output='textual content marketing the event',
    agent=marketing_agent,
    output_file='marketing_report.md',
    async_execution=True,
    verbose=True,
)
final_task=Task(
    description='Consolidate venue, logistics, and marketing outputs into a final event plan.',
    expected_output='A single consolidated final event plan.',
    agent=ven_coor,
)
crew=Crew(
    agents=[ven_coor,log_man,marketing_agent],
    tasks=[ven_coordinator,log_task,mark_task,final_task],
    verbose=False
)
inputs={
    'event':'Wedding',
    'country':'india',
    'budget':5000000,
}
result=crew.kickoff(inputs=inputs)
import json
from pprint import pprint

with open('Venue_details.json') as f:
   data = json.load(f)

pprint(data)
try:
    from IPython.display import Markdown, display
    with open('marketing_report.md') as f:
        display(Markdown(f.read()))
except ImportError:
    with open('marketing_report.md') as f:
        print(f.read())
