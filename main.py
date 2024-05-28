import streamlit as st
import feedparser
import requests
from bs4 import BeautifulSoup, SoupStrainer
import openai
import os
import re
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.prompts import ChatPromptTemplate

# Set your API keys
openai.api_key = os.environ['oai']
lc_key = os.environ['langchain']

# Setup LangSmith for token usage monitoring :
# Setup LANGSMITH for LLMOps :
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "NewsLetter"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = lc_key

st.title("RSS Feed Newsletter Aggregator & Rewriter")

llm = ChatAnthropic(model="claude-3-opus-20240229",
                    api_key=os.environ['claude'])

with open('prompt.txt', 'r') as file:
    prompt = ChatPromptTemplate.from_template(file.read())

chain = {'feed1': RunnablePassthrough()} | prompt | llm | StrOutputParser()

def download_newsletter(text, html):
    with open('rewritten_newsletter.txt', 'w') as file:
        file.write(text)
    with open('rewritten_newsletter.html', 'w') as file:
        file.write(html)
    st.markdown(f"Download [rewritten_newsletter.txt](rewritten_newsletter.txt)")
    st.markdown(f"Download [rewritten_newsletter.html](rewritten_newsletter.html)")
    return

def extract_html_content(text):
    """
    Extracts HTML content from a given string.

    Parameters:
    text (str): The input string containing HTML content.

    Returns:
    str: The extracted HTML content.
    """
    # Find the starting position of the HTML content
    html_start = text.find("<html>")

    # Extract the HTML content
    if html_start != -1:
        html_content = text[html_start:]
    else:
        html_content = ""

    return html_content


# Get URLs from user
urls_input = st.text_area("Enter URLs (one per line):")

# Process URLs
if st.button("Generate Newsletter"):
    urls = [url.strip() for url in urls_input.split("\n") if url.strip()]

    articles = []
    for url in urls:
        try:
            feed = feedparser.parse(url)
            if feed.bozo == 1 or not feed.entries:
                # Treat as regular URL
                response = requests.get(url)
                soup = BeautifulSoup(response.content, 'html.parser')
                paragraphs = soup.find_all('p')
                article_text = ' '.join([p.get_text() for p in paragraphs])
                articles.append({
                    'title': url,
                    'text': article_text
                })
            else:
                # Treat as RSS feed
                latest_entry = feed.entries[0]
                response = requests.get(latest_entry.link)
                soup = BeautifulSoup(response.content, 'html.parser')
                paragraphs = soup.find_all('p')
                article_text = ' '.join([p.get_text() for p in paragraphs])
                articles.append({
                    'title': latest_entry.title,
                    'text': article_text
                })
        except Exception as e:
            st.write(f"Failed to process {url}: {e}")

    if not articles:
        st.write("No articles found. Please check your URLs.")
        st.stop()

    # Combine article text into a single string
    combined_text = "\n\n".join(
        [f"**{a['title']}**\n{a['text']}" for a in articles])

    if len(combined_text) < 100:
        st.write(
            "The combined text from the articles is too short. Please ensure that the articles have sufficient content."
        )
        st.stop()

    try:
        rewritten_newsletter = chain.invoke({"feed1": combined_text})
        st.write("### Newsletter")

        # Create HTML version of the newsletter
        newsletter_html = BeautifulSoup(rewritten_newsletter, 'html.parser').prettify()

        content = extract_html_content(newsletter_html)

        st.markdown(content, unsafe_allow_html=True)

        print(rewritten_newsletter)

        if rewritten_newsletter:
            st.download_button(label= "Download Newsletter", data = content, file_name = "Newsletter.txt", mime = "text/plain")
    except Exception as e:
        st.write(f"Failed to generate newsletter: {e}")
        st.stop()

