## Writo: The Newsletter Writing Assistant

![Writo Logo](images/writo.jpg)

Writo is a powerful newsletter writing assistant that aggregates and rewrites content from RSS feeds using advanced language models. The application is built with Streamlit, making it easy to set up and run locally. Writo leverages the capabilities of the LangChain and OpenAI frameworks to provide high-quality rewritten newsletters.

### Features

- Aggregates content from multiple RSS feeds
- Uses advanced language models to rewrite the aggregated content
- Simple and intuitive user interface with Streamlit
- Downloadable rewritten newsletters

### Prerequisites

- Python 3.7 or higher
- Streamlit
- Feedparser
- Requests
- BeautifulSoup
- OpenAI
- LangChain

### Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/yourusername/writo.git
    cd writo
    ```

2. Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

3. Set up your environment variables:

    ```bash
    export oai=YOUR_OPENAI_API_KEY
    export langchain=YOUR_LANGCHAIN_API_KEY
    export claude=YOUR_CLAUDE_API_KEY
    ```

### Usage

1. Create a file named `prompt.txt` with the prompt template for the language model. Example content for `prompt.txt`:

    ```
    You are a helpful assistant whose main aim is to assist the user in rewriting the aggregated newsletter content to be engaging and informative. If you don't know, just say you don't know. Think step by step before answering any question. {feed1}
    ```

2. Run the Streamlit application:

    ```bash
    streamlit run main.py
    ```

3. Open your web browser and go to `http://localhost:8501`.

4. Enter the RSS feed URLs (one per line) in the provided text area and click "Generate Newsletter".

5. The aggregated and rewritten newsletter content will be displayed on the page. You can download the rewritten newsletter as a text file.


### Contributing

Contributions are welcome! Please open an issue or submit a pull request if you have any improvements or suggestions.

### License

This project is licensed under the MIT License.

---

Enjoy using Writo to create engaging newsletters effortlessly!