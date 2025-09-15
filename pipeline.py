# Assignment1.py

import argparse
from langchain.chat_models import init_chat_model
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List
import mlflow
import time
from langchain.schema.runnable import RunnableLambda, RunnableMap
import json
from ddgs import DDGS

from dotenv import load_dotenv
import os

# Load .env file
load_dotenv()


def mlflow_span(name, fn, parent_run_id):
    """Decorator to wrap a chain/function call into an MLflow nested run."""
    def wrapper(inputs):
        with mlflow.start_run(experiment_id="839027490218794265", run_name=name, nested=True, parent_run_id=parent_run_id):
            start = time.time()
            result = fn(inputs)
            duration = time.time() - start
            print(result)
            # log inputs/outputs safely
            if isinstance(result, BaseModel):
                mlflow.log_dict(result.model_dump(), f"{name}_output.json")
            elif isinstance(result, dict):
                mlflow.log_dict(result, f"{name}_output.json")
            elif isinstance(result, str):
                mlflow.log_param(f"{name}_output", result)
            elif hasattr(result, "content"):
                try:
                    parsed = json.loads(result.content)  # try JSON parse
                    mlflow.log_dict(parsed, f"{name}_output.json")
                except Exception:
                    # fallback: save plain text
                    mlflow.log_text(result, f"{name}_output.txt")
            mlflow.log_metric("duration_sec", duration)
            return result
    return wrapper


model_name='gemini-2.0-flash'


model = init_chat_model(model_name, model_provider="google_genai")


class StockCodeSchema(BaseModel):
    company_name: str = Field(description="Name of the Company given as input")
    stock_code: str = Field(description="Stock code of the Company")

stock_parser = PydanticOutputParser(pydantic_object=StockCodeSchema)

stock_prompt = PromptTemplate(
    template="""
You are a financial assistant.
Given the company name: "{company_name}", get its stock ticker symbol
used in Yahoo Finance or NASDAQ/NYSE and return a structured JSON with this exact schema.

{format_instructions}
""",
    input_variables=["company_name"],
    partial_variables={"format_instructions": stock_parser.get_format_instructions()},
)


ticker_chain = stock_prompt | model | stock_parser


class NewsSummary(BaseModel):
    news_article: str
    news_summary: str

news_summary_parser = PydanticOutputParser(pydantic_object=NewsSummary)

news_summary_prompt = PromptTemplate(
    template="""
You are a News Summarizer.
Given this news article: "{news_article}", return its summary and key findings in 3 sentences.

{format_instructions}
""",
    input_variables=["news_article"],
    partial_variables={"format_instructions": news_summary_parser.get_format_instructions()},
)

def get_news_summary(news_article: str):
    chain_prompt = news_summary_prompt.format_prompt(news_article=news_article)
    response = model.invoke(chain_prompt.to_messages())
    parsed = news_summary_parser.parse(response.content)
    return parsed.news_summary

def get_news_duckduckgo(company_name, max_results=5):
    with DDGS() as ddgs:
        results = [r for r in ddgs.news(company_name, max_results=max_results)]
    news_list = [{"title": r["title"], "link": r["url"], "source": r["source"], "date": r["date"], "summary": get_news_summary(r["body"])} for r in results]
    return ";".join([i["summary"] for i in news_list])


# Schema for Sentiment Output
class SentimentSchema(BaseModel):
    company_name: str = Field(description="Name of the Company given as input")
    stock_code: str = Field(description="Stock code of the Company given as input")
    newsdesc: str = Field(description="News related to the Company given as input")
    sentiment: str = Field(description="Overall sentiment (Positive, Negative, Neutral).")
    people_names: List[str] = Field(description="All names of people mentioned.")
    places_names: List[str] = Field(description="All location names (cities, countries, regions)")
    other_companies_referred: List[str] = Field(description="Other companies explicitly mentioned.")
    related_industries: List[str] = Field(description="Industries/sectors relevant to the news.")
    market_implications: str = Field(description="1–2 sentences of the likely market effect.")
    confidence_score: float = Field(description="0–1 reflecting confidence in this analysis")


sentiment_parser = PydanticOutputParser(pydantic_object=SentimentSchema)


prompt_sentiment = PromptTemplate(
    template="""
You are a financial news sentiment analyzer.
Given the following news headlines about {company_name} ({stock_code}),
extract and return a structured JSON with this exact schema:

{format_instructions}

Details:
- "sentiment": Overall sentiment (Positive, Negative, Neutral).
- "people_names": Extract all names of people mentioned.
- "places_names": Extract all location names (cities, countries, regions).
- "other_companies_referred": List other companies explicitly mentioned.
- "related_industries": List industries/sectors relevant to the news.
- "market_implications": Summarize in 1–2 sentences the likely market effect.
- "confidence_score": Float 0–1 reflecting confidence in this analysis.

News:
{news}
""",
    input_variables=["company_name", "stock_code", "news"],
    partial_variables={"format_instructions": sentiment_parser.get_format_instructions()}
)



sentiment_chain = prompt_sentiment | model | sentiment_parser


def sentiment_pipeline(company_name: str):
    mlflow.set_tracking_uri("http://20.75.92.162:5000/")
    with mlflow.start_run(experiment_id="839027490218794265", run_name=f"Sentiment_{company_name}") as parent:
        parent_run_id  = parent.info.run_id
        ticker_chain_ml = RunnableLambda(mlflow_span("StockCodeExtraction", ticker_chain.invoke, parent_run_id))
        news_chain_ml = RunnableLambda(mlflow_span("NewsFetching", get_news_duckduckgo, parent_run_id))
        sentiment_chain_ml = RunnableLambda(mlflow_span("SentimentParsing", sentiment_chain.invoke, parent_run_id))

        pipeline = RunnableMap({
            "company_name": lambda x: x,        # passthrough
            "stock_code": ticker_chain_ml,         # LLM chain
            "news": news_chain_ml,
        }) | sentiment_chain_ml


        result = pipeline.invoke(company_name)
        mlflow.log_param("company_name", company_name)
        mlflow.log_dict(result.model_dump(), "sentiment_output.json")
        return result.model_dump()


def main():
    parser = argparse.ArgumentParser(description="Run Stock Sentiment Pipeline")
    parser.add_argument(
        "company_name",
        type=str,
        help="Company name to analyze (eg: 'Tesla', 'Apple')")
    args = parser.parse_args()

    sentiment_pipeline(args.company_name)


if __name__ == "__main__":
    main()
