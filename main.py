import streamlit as st
import feedparser
import requests
from bs4 import BeautifulSoup
import openai
import os
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
                    api_key=os.environ['claude']

with open('prompt.txt', 'r') as file:
    prompt = ChatPromptTemplate.from_template(file.read())

chain = {'feed1': RunnablePassthrough()} | prompt | llm | StrOutputParser()

def download_newsletter(text):
    with open('rewritten_newsletter.txt', 'w') as file:
        file.write(text)
        st.markdown(f"Download [rewritten_newsletter.txt](rewritten_newsletter.txt)")
    return


# Get RSS feed URLs from user
rss_urls_input = st.text_area("Enter RSS feed URLs (one per line):")
# Process RSS feed URLs
if st.button("Generate Newsletter"):
    rss_urls = [
        url.strip() for url in rss_urls_input.split("\n") if url.strip()
    ]

    articles = []
    for url in rss_urls:
        try:
            feed = feedparser.parse(url)
            if not feed.entries:
                st.write(f"No entries found for {url}")
                continue
            latest_entry = feed.entries[0]

            # Use BeautifulSoup for content extraction
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
        st.write("No articles found. Please check your RSS feed URLs.")
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
        st.write(rewritten_newsletter)
        if rewritten_newsletter:
            st.button("Download Newsletter", on_click=download_newsletter, args=(rewritten_newsletter,))
    except Exception as e:
        st.write(f"Failed to generate newsletter: {e}")
        st.stop()
