import warnings
warnings.filterwarnings('ignore')

from crewai import Crew,Agent,Task,LLM

import os
api_key=os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise RuntimeError("API key not entered in terminal")
llm=LLM(
    model="gemini/gemini-2.5-flash",
    api_key=api_key
)
support_agent=Agent(
    role="Customer Support agent",
    goal="to provide support to customers by answering their enquiry",
    backstory="You are a senior customer support agent working with crewai company.You provide support to the customers and guide them through issues they face",
    allow_delegation=False,
    #verbose=True,
    llm=llm
)
quality_checker=Agent(
    role="Quality resolution expert",
    goal="to check the quality of resolution provided by support_agent",
    backstory="You are a Query resolution quality expert at crewai.Ensure that the answer given by support_agent is appropraite and answers every query raised by {customer} . Maintain friendly and formal tone.",
    allow_delegation=True,
    #verbose=True,
    llm=llm
)
from crewai_tools import SerperDevTool,ScrapeWebsiteTool,WebsiteSearchTool
docs_scrape_tool=ScrapeWebsiteTool(
    website_url="https://docs.crewai.com/"
)
#define tasks
inquiry_res=Task(
    description="customer-{customer} has raised an inquiry-'{inquiry}'.Respond to the customer's inquiry.Keep the answer in short(2-3 paragraphs) and simple and start by addresing the customer",
    expected_output="A detailed answer to the {inquiry}",
    tools=[docs_scrape_tool],
    agent=support_agent,
    verbose=False
)
quality_ass=Task(
    description="Go through the inquiry resolution provided by support agent and ensure every question is answered",
    expected_output="A detailed answer to the inquiry raised by the customer",
    agent=quality_checker,
        context=[inquiry_res],
        verbose=False

)
crew=Crew(
    agents=[support_agent,quality_checker],
    tasks=[inquiry_res,quality_ass],
    verbose=True,
    memory=True
)
inputs={
    "customer":"abhinav",
    "inquiry":"What is crewai?"
}
result=crew.kickoff(inputs=inputs)
print(result)