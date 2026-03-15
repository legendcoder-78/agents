import warnings 
warnings.filterwarnings('ignore')

from crewai import Agent,Task,Crew,LLM

import os
from utils import GEMINI_API_KEY, serper_api_key
api_key = GEMINI_API_KEY()
llm=LLM(
    model='gemini/gemini-2.5-flash',
    api_key=api_key
)
os.environ["SERPER_API_KEY"]=serper_api_key()
#Agent 1
sales_rep=Agent(
    role="Sales representative Agent",
    goal="go through the web and know about the company or organisation, its services and products. Then genearate an ideal customer profile for the services and products offered our  company",
    backstory="You are a senior sales representative agent working for the comapany {company}.Your job is to generate an ideal customer profile for our company {company}, so that the company can get to know which are the best servise and products that can be sold",
    allow_delegation=False,
    verbose=True,
    llm=llm,
)
#Agent 2
lead_sales_rep=Agent(
    role="Lead sales representative",
    goal='You take details from sales representative and generate compelling messages for the ideal customer so that he takes interest in our products and services',
    backstory='You are lead sales representative working in company {company}.Your job is to reach out to potential customers and convince them to use our services and products.',
    llm=llm,
)
#tools
from crewai_tools import DirectoryReadTool,FileReadTool,SerperDevTool
directory_read_tool=DirectoryReadTool()
file_read_tool=FileReadTool()
search_tool=SerperDevTool()
#Creating a custom tool to analyse the sentiment of data
from crewai.tools import BaseTool
class SentimentAnalysisTool(BaseTool):
    name:str='Sentiment Analysis Tool'
    description:str='This tool analyses ' \
    '' \
    'the sentiment of text and returns whether it is positive , negative or neutral'
    def _run(self,text:str)->str:
        #We write our custom code here
        return 'positive'
sentiment_analysis_tool=SentimentAnalysisTool()

#now define the tasks
profile_analyser=Task(
    description='lead {lead_name} from sector {sector} has shown interest in our company{company}.Go through {lead_name} profile.Prepare a detailed report of the potential client{lead_name} regarding the services he may need from our company, by going through {lead_name}  requirements',
    expected_output='Products and services that {lead_name} maybe  interested in',
    tools=[directory_read_tool,search_tool,file_read_tool],
    agent=sales_rep,
)
outreach_task=Task(
    description='frame emails addressing the lead {lead_name} from {ceo}, convincing him to use the services and products predicted in profile_analyser task and provided by us',
    expected_output='convincing emails to  {lead_name},to use our services an products',
    tools=[sentiment_analysis_tool,search_tool],
    agent=lead_sales_rep,
)
crew=Crew(agents=[sales_rep,lead_sales_rep],
          tasks=[profile_analyser,outreach_task],
          verbose=True,
          memory=True,
          )
inputs={
    'company':'Google',
    'lead_name':'DeepLearning.AI',
    'ceo':'Abhinav',
    'sector':'Online Education platform',
}
results=crew.kickoff(inputs=inputs)
print(results)

