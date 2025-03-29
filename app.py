from crewai import Agent, Task, Crew, LLM
from crewai_tools import SerperDevTool
from dotenv import load_dotenv

load_dotenv()

topic="Medical Industry using Generative AI"

# Tool 1- Basic configuration
llm = LLM(model="gpt-3.5-turbo")

# Tool 2-
search_tool= SerperDevTool(n=10)

# Agent 1
senior_research_analyst= research_agent = Agent(
    role="Senior Research Analyst",
    goal="Research, analyze, and synthesize comprehensive information on {topic} from reliable web sources ",
    backstory="You're an expert research analyst with advanced web research skills."
               "You excel at finding, analyzing, and synthesizing information from"
               "across the internet using search tools. You're skilled at distinguishing reliable sources from unreliable ones,"
               "fact-checking, cross-references, and identifying key patterns and insights."
               "You provide well-organized research briefs with proper citations and score verifications. Your"
               "analysis includes both raw data and interpreted insights, making complex information"
               "accessible and actionable.",
    allow_delegation= False,
    tools=[search_tool],
    verbose=True,
    llm=llm
)

#  Agent 2- Content Writer
content_writer= Agent(
    role="Content Writer",
    goal="Transform research findings into engaging blog posts while maintaining accuracy",
    backstory="You're a skilled content writer specialized in creating"
              "engaging, accessible content from technical research."
              "you work closely with the senior research analyst and excel at maintaining the perfect balance"
              "between informative and entertaining writing, "
              "while ensuring all facts and citations from the research are properly incorporated."
              "You have a talent for making complex topics approachable without oversimplifying them.",
    allow_delegation= False,
    verbose=True,
    llm=llm
)

# Research Tasks
research_tasks= Task(
    description= ("""
        1. Conduct comprehensive research on {topic} including:
         - Recent development and news
         - Key industry trends and innovations
         - Expert opinions and analyzes
         - Statistical data and market insights
        2. Evaluate source credibility and fact-check all information
        3. Organize findings into a structured research brief
        4. Include all relevant citations and sources
    """),
    expected_output=""" A detailed research report containing:
    - Executive summary of key findings
    - Comprehensive analysis of current trends and developments
    - list of verified facts and statistics
    - All citations and links to original sources
    Please format with clear sections and bullet points for easy references.
    """,
    agent= senior_research_analyst
)


# Advanced configuration with detailed parameters
llm = LLM(
    model="gpt-3.5-turbo",
    temperature=0.7,        # Higher for more creative outputs
    timeout=120,           # Seconds to wait for response
    max_tokens=4000,       # Maximum length of response
    top_p=0.9,            # Nucleus sampling parameter
    frequency_penalty=0.1, # Reduce repetition
    presence_penalty=0.1,  # Encourage topic diversity
    response_format={"type": "json"},  # For structured outputs
    seed=42               # For reproducible results
)