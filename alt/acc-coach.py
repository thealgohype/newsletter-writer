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
from interpreter import interpreter
from openai import OpenAI

client = OpenAI(api_key=os.environ['oai'])
lc_key = os.environ['langchain']

#Setup LangSmith for token usage monitoring :
# Setup LANGSMITH for LLMOps :
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "AccountingCoach"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = lc_key


# Sidebar for model selection and file upload
st.sidebar.title("Configuration")
model = st.sidebar.selectbox("Select the model", [
    "gpt-3.5-turbo", "gpt-4", "claude-sonnet", "claude-opus", "google-gemini","llama-3"
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

# Display branding
st.sidebar.divider()
with st.sidebar.expander(label="Created By:", expanded=True):
    st.sidebar.image("https://lh3.googleusercontent.com/v1fRBZY3mv3MzVmlWWEOU2VSCKpqgppBriaOrjX4FyEqLf2hKNOhcu1kWhjQAXmzD9HlmlQEWs-qIkRa7nbaZzMwO28=w128-h128-e365-rj-sc0x00ffffff")

# Title and description
st.title("Accounting Coach Content Generator")
st.markdown("## Certificate Question Generator")

# Process uploaded file
if uploaded_file:
    try:
        if uploaded_file.name.endswith('.pdf'):
            st.write("### Data Summary")

        # Define prompt template
        prompt_template = "You are an AI specialized in data analysis. Using the following dataset, provide detailed analysis, actionable insights, and generate relevant plots and graphs. {data}"
        prompt = ChatPromptTemplate.from_template(prompt_template)

        # Chain setup
        chain = {'data': RunnablePassthrough()} | prompt | llm | StrOutputParser()

        # Generate analysis
        st.write("### Analysis and Insights")
        analysis = chain.invoke({'data': data.to_dict()})
        st.write(analysis)

        # Generate visualizations
        st.write("### Visualizations")
        st.write("#### Correlation Matrix")
        corr = data.corr()
        sns.heatmap(corr, annot=True, cmap='coolwarm', linewidths=0.5)
        st.pyplot(plt.gcf())

        # Additional plots (example: distribution plots for numerical columns)
        st.write("#### Distribution Plots")
        num_cols = data.select_dtypes(include='number').columns
        for col in num_cols:
            plt.figure()
            sns.histplot(data[col], kde=True)
            st.pyplot(plt.gcf())

    except Exception as e:
        st.error(f"Error processing file: {e}")
else:
    st.info("Please upload an Excel or CSV file to proceed.")


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

if uploaded_file and len(uploaded_file) == 2 and all([file.name.endswith('.pdf') for file in uploaded_file]):
    content1 = uploaded_file[0].read()
    content2 = uploaded_file[1].read()

    prompt = "You are an AI expert. Please create certificate questions based on the following two attachments and the material provided therein."

    chat_client = ChatAnthropic(model="your_model_name_here", api_key="your_api_key_here")

    completion = chat_client.generate_completion(prompt, content1, content2)

    st.write(completion)
else:
    st.info("Please upload two PDF files to proceed.")