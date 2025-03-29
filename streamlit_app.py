import streamlit as st
from crewai import Agent, Task, Crew, LLM
from crewai_tools import SerperDevTool
from dotenv import load_dotenv
import os 

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SERPER_API_KEY = os.getenv("SERPER_API_KEY")

# Streamlit UI
st.title("📖 ScholarlyAI - Your Automated Blog Creator")
topic = st.text_input("Enter a topic:")

if st.button("Generate Blog Post") and topic:
    with st.spinner("Generating content..."):
        
        # LLM Configuration
        llm = LLM(
            model="gpt-3.5-turbo",
            api_key=OPENAI_API_KEY,
            temperature=0.7,
            timeout=120,
            max_tokens=4000,
            top_p=0.9,
            frequency_penalty=0.1,
            presence_penalty=0.1,
            response_format={"type": "json"},
            seed=42
        )

        # Search Tool
        search_tool = SerperDevTool(api_key=SERPER_API_KEY, n=10)

        # Agents
        research_agent = Agent(
            role="Senior Research Analyst",
            goal=f"Research and analyze comprehensive information on {topic}",
            backstory="Expert in web research and information synthesis.",
            allow_delegation=False,
            tools=[search_tool],
            verbose=True,
            llm=llm
        )

        content_writer = Agent(
            role="Content Writer",
            goal=f"Create an engaging blog post on {topic}",
            backstory="Skilled writer transforming research into accessible content.",
            allow_delegation=False,
            verbose=True,
            llm=llm
        )

        # Tasks
        research_task = Task(
            description=f"Conduct comprehensive research on {topic} and compile key insights.",
            expected_output="A structured research brief with citations.",
            agent=research_agent
        )

        writing_task = Task(
            description="Write a well-structured blog post based on the research findings.",
            expected_output="A formatted blog post in markdown with proper citations.",
            agent=content_writer
        )

        # Crew Execution
        crew = Crew(
            agents=[research_agent, content_writer],
            tasks=[research_task, writing_task],
            verbose=True
        )
        
        result = crew.kickoff(inputs={"topic": topic})
        
        st.subheader("Generated Blog Post:")
        st.markdown(result, unsafe_allow_html=True)
