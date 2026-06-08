import streamlit as st
import os
import requests
from bs4 import BeautifulSoup
from crewai.tools import tool
from crewai import Agent, Task, Crew, LLM
from dotenv import load_dotenv

load_dotenv()

openrouter_llm = LLM(
  model="openrouter/meta-llama/llama-3.3-70b-instruct",
    api_key=os.environ.get("OPENROUTER_API_KEY")
)
st.set_page_config(page_title = "AI Market Intelligence", page_icon="🤖")
st.title("🤖 AI Market Intelligence Agent")
target_url = st.text_input("Enter competitor's website URL:", "https://zomato.com/blog")

@tool("Enterprise web scraper")
def scrape_site(url):
    """useful to scrape web content"""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        clean_text = soup.get_text()
        return clean_text
    else:
        return f"Failed to retrieve content. Status code: {response.status_code}"

intelligence_agent = Agent(
  role="Senior Corporate Intelligence Analyst",
    goal="Scrape competitor websites to extract core product offerings and features.",
    backstory="You are an expert at dissecting competitor landing pages to identify exactly what they are selling.",
    tools = [scrape_site],
    llm = openrouter_llm,
    verbose = True
)

strategy_agent = Agent(
  role="Chief Product Strategist",

  goal="Analyze competitor data and derive actionable business strategies.",

  backstory="You take raw data about competitors and find market gaps, threats, and counter-strategies.",

  llm=openrouter_llm,

  verbose=True
)
linkedin_agent = Agent(
    role="Social Media Marketing Specialist",
    goal="Convert complex business reports into viral, engaging LinkedIn posts.",
    backstory="You are an expert content creator who knows how to write catchy, professional tech updates using bullet points and emojis to get high engagement.",
    llm=openrouter_llm,
    verbose=True
)
if st.button("Start Intelligent Analysis"):
  st.write(f"Analyzing {target_url}... please wait")
  extraction_task = Task(
    description=f"Use your scraping tool to scan {target_url} and find the latest news updates.",
    expected_output="A list of recent updates found on the website.",
    agent=intelligence_agent
)
  strategy_task = Task(
    description="Take the competitor updates list and analyze the market impact. Outline 1 major market threat and 2 counter-strategies we should adopt to stay ahead.",
    expected_output="A professional Competitor Analysis Report in clean Markdown format with sections: Market Threats and Recommended Strategies.",
    agent=strategy_agent,
    output_file="Competitor_Strategy_Report.md"
)
  linkedin_task = Task(
    description="Read the generated Competitor Analysis Report and summarize it into an eye-catching LinkedIn post to showcase our business strategy skills.",
    expected_output="A high-engagement LinkedIn post featuring a catchy hook, 3 clear bullet points, professional emojis, and relevant hashtags.",
    agent=linkedin_agent
)
  enterprise_crew = Crew(
    agents=[intelligence_agent, strategy_agent, linkedin_agent],
    tasks=[extraction_task, strategy_task, linkedin_task]
)

  result = enterprise_crew.kickoff()

  st.success("Analysis Complete!")
  st.markdown("### 📊 Final Report")
  st.markdown(result.raw)