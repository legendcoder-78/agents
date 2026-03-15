import warnings
warnings.filterwarnings('ignore')
from crewai import Agent,Task,Crew,Process,LLM
from utils import GEMINI_API_KEY,serper_api_key
api_key=GEMINI_API_KEY()
serper_api_key=serper_api_key()
from crewai_tools import SerperDevTool,ScrapeWebsiteTool
search_tool=SerperDevTool()
scrape_tool=ScrapeWebsiteTool()
llm=LLM(
    model='gemini/gemini-2.5-flash',
    api_key=api_key
)
#Agent 1:Data Analyst
data_analyst=Agent(
    role='Data Analyst',
    goal='To fetch market financial data ,clean it and process it.You then compute technical indicators like moving averages and identify trends and patterns. You then genrate a structured market analysis report',
    backstory='You are a Data analyst.By processing recent market data and analysing it you help the other agents take actions with respect to investment related decisions',
    verbose=True,
    allow_delegation=False,
 llm=llm,
 Tools=[search_tool,scrape_tool]
)
trading_strategy_agent=Agent(
    role='Trading Strategy Specialist',
    goal='Interpret the data analyst report and decide what trade to take',
    backstory='You are a trading strategy agent. You convert analysis done by data analyst agent into a trading strategy and decide what trade to take by choosing a trading strategy and defining the rules.',
    verbose=True,
    allow_delegation=False,
    llm=llm,
     Tools=[search_tool,scrape_tool]

)
Risk_Management_Agent=Agent(
    role="Risk Management Agent",
    goal='Interpret the trading strategy developed by Trading strategy agent and decide whether the trade is safe enough',
   backstory='You are a Risk Management Agent.You Evaluate the risk of trading before execution. You find parameters like Position size,Risk/ reward ratio etc. and enforce rules like max risk per trade or setting diversification limits',
verbose=True, allow_delegation=False,  Tools=[search_tool,scrape_tool],
llm=llm
)
execution_agent=Agent(
    role='Execution Agent',
    goal='You actually place the trade after all other agents have done their parts in the process.',
    backstory='You perform the final action in the trading process.You generate the approved and final strategy,take decisions like buying and selling orders, tracking order status and logging trade details',
    verbose=True, allow_delegation=False,
     Tools=[search_tool,scrape_tool],
llm=llm
)
#Tasks
analysis=Task(
    description='fetch stock market data,retrieve stock price history,fetch trading volumes and get market indicators',
    expected_output='a detailed report of market analysis done',
    agent=data_analyst
)
making_strategy=Task(
    description='analyse the report generated in analysis task and frame a trading strategy based on the report by identifying trends,calculating moving averages and searching trade strategies',
    expected_output='A textual content that explains about the trading strategy developed ',
    agent=trading_strategy_agent
)
risk_management=Task(
    description='Analyse the trading strategy framed in making_strategy task and check whether the trades are safe and controlled',
    expected_output='modified trading strategy after risk management',
    agent=Risk_Management_Agent
)
execution=Task(
    description='Generate the final approved trade plan',
    expected_output='a detailed plan of trading',
        agent=execution_agent,
output_file='final_trade_strategy.txt',
)
from langchain_google_genai import ChatGoogleGenerativeAI
manager_llm=ChatGoogleGenerativeAI(
    model='gemini/gemini-2.5-flash',
    temperature=0.7
)

crew=Crew(
    agents=[data_analyst,trading_strategy_agent,Risk_Management_Agent,execution_agent],
    tasks=[analysis,making_strategy,risk_management,execution],
    verbose=True,
    manager_llm=manager_llm,
    process=Process.hierarchical,
)
results=crew.kickoff()
from IPython.display import Markdown
Markdown(results)