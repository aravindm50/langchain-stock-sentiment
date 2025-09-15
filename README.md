# 🚀 LangChain Stock Sentiment Pipeline

This project builds a **LangChain-powered pipeline** that:  
1. Accepts a company name as input.  
2. Extracts or generates its stock code with LLM.  
3. Fetches recent news using **DuckDuckGo Search** (`duckduckgo-search` / `ddgs`).  
4. Summarizes and analyzes sentiment with **Google Gemini-2.0-flash** (via Vertex AI) or another LLM.  
5. Outputs a structured **JSON profile** with sentiment, entities, and market implications.  
6. Uses **MLflow** for tracing, sub-run monitoring, and debugging prompts.  

---

## 📦 Tech Stack
- **Framework:** LangChain  
- **LLM:** Google Gemini-2.0-flash (via Vertex AI)  
- **News Source:** DuckDuckGo Search (no API key required)  
- **Ticker Lookup:** LLM - Gemini 2.0-Flash
- **Prompt Monitoring & Observability:** MLflow  
- **Language:** Python 3.10+  

---

## ⚙️ Installation

Clone repo and install dependencies:

```bash
git clone https://github.com/aravindm50/langchain-stock-sentiment.git
cd langchain-stock-sentiment

# Mlflow experiment details:

# Set the Mlflow tracking URI in the pipeline.py file
mlflow.set_tracking_uri("your mlflow tracking uri")

# Set the experiment ID in the pipeline.py file
experiment_id="your experiment id"

# Recommended to use venv or conda
pip install -r requirements.txt

```

## 🔑 Configuration
1. **MLflow**
    - Start MLflow tracking server locally (default: http://127.0.0.1:5000):
    - This opens MLflow dashboard at http://127.0.0.1:5000

2. **Google Gemini (Vertex AI)**
    -   Set environment variables in .env file in the same folder:
``` bash
GOOGLE_CLOUD_PROJECT="your-gcp-project-id"
GOOGLE_API_KEY=""
GOOGLE_GENAI_USE_VERTEXAI=True
```

# ▶️ Running the Pipeline
    
**Example**: Run for Tesla

``` bash
python pipeline.py "Tesla"
```
