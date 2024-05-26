import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.document_loaders import RSSFeedLoader
from interpreter import interpreter
from openai import OpenAI

client = OpenAI(api_key=os.environ['oai'])
lc_key = os.environ['langchain']

#Setup LangSmith for token usage monitoring :
# Setup LANGSMITH for LLMOps :
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "NewsLetter"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = lc_key


# Sidebar for model selection and file upload
st.sidebar.title("Configuration")
model = st.sidebar.selectbox("Select the model", [
    "gpt-3.5-turbo", "gpt-4", "claude-sonnet", "claude-opus", "google-gemini","llama-3", "cohere"
], placeholder="Select Model")

uploaded_file = st.sidebar.file_uploader("Upload your Explanation and CheetSheet", type=["pdf", "txt"])

if model == "gpt-3.5-turbo":
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7, api_key=os.environ['oai'])
elif model == "gpt-4":
    llm = ChatOpenAI(model="gpt-4o", temperature=0.7, api_key=os.environ['oai'])
elif model == "claude-sonnet":
    llm = ChatAnthropic(model="claude-3-sonnet-20240229", api_key=os.environ['claude'])
elif model == "claude-opus":
    llm = ChatAnthropic(model="claude-3-opus-20240229", api_key=os.environ['claude'])
elif model == "google-gemini":
    llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=os.environ['gemini'])
elif model == "llama-3":
    llm = ChatGroq(model="llama3-70b-8192", groq_api_key=os.environ['groq'])
elif model == "cohere":
    llm= ChatCohere(model="command", cohere_api_key=os.environ['cohere'])



# Display branding
st.sidebar.divider()
with st.sidebar.expander(label="Created By:", expanded=True):

    st.sidebar.image("https://lh3.googleusercontent.com/v1fRBZY3mv3MzVmlWWEOU2VSCKpqgppBriaOrjX4FyEqLf2hKNOhcu1kWhjQAXmzD9HlmlQEWs-qIkRa7nbaZzMwO28=w128-h128-e365-rj-sc0x00ffffff")

# Title and description
st.title("Newsletter Generator")
st.markdown("## Give us inspiration to craft the perfect Newsletter for you.")

        # Define prompt template
prompt_template = '''

You are an AI assistant named NewsletterGPT. Your purpose is to generate high-quality, engaging newsletters based on the latest articles from 2-3 RSS feeds provided by the user.

When the user inputs their selected RSS feeds, carefully analyze the content of the most recent articles from each feed. Identify the key topics, themes, insights, and newsworthy information across the articles.

Use this analysis to write an original newsletter that synthesizes the most important and interesting information from the RSS articles into a cohesive, well-organized piece of content. The newsletter should be written in a professional yet engaging style appropriate for the target audience.

Key requirements:

The newsletter should be around 800-1200 words in length, with a clear introduction, body organized into sections by theme/topic, and conclusion.
Focus on the most noteworthy and unique information and insights from across the RSS articles. Do not simply summarize the articles.
Cite the RSS sources inline where appropriate using hyperlinks.
Use clear formatting with headings, subheadings, short paragraphs, bullet lists where appropriate to make the newsletter skimmable and visually appealing.
Ensure the writing quality is excellent - engaging, error-free, and easy to read.
Adapt the writing style and content to be well-suited for the target audience and subject matter of the RSS feeds.
Remember, the user is counting on you to generate a newsletter they will be proud to send to their audience. Take the time to craft a thoughtful, polished, and insightful piece of content that offers unique value, based on the specific RSS articles provided. Let me know if you have any other questions!

Feed : 
{feed}
'''
prompt = ChatPromptTemplate.from_template(prompt_template)
# Chain setup
chain = {'feed': RunnablePassthrough()} | prompt | llm | StrOutputParser()

# Generate analysis
st.write("### Analysis and Insights")
analysis = chain.invoke({'feed':  })
st.write(analysis)
st.divider()



'''
assistant = client.beta.assistants.create(
  name="Data Analyst Bot",
  instructions="You are an amzing data scientist and data analyst with a knack for finding insights in data and creating perfect visualisations to explain data trends. Write and run code to alter datasets and generate charts for generating a detailed and insightful data analysis reports.",
  tools=[{"type": "code_interpreter"}],
  model="gpt-4o",
)

thread = client.beta.threads.create()

message = client.beta.threads.messages.create(
  thread_id=thread.id,
  role="user",
  content=st.text_input("please ask the data a question")
)'''
