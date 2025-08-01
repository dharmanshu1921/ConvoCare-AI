import os
import json
import requests
from langchain_community.document_loaders import TextLoader, PyPDFLoader, CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.tools import tool
from dotenv import load_dotenv
from langdetect import detect
import logging
from qdrant_client import QdrantClient
from langchain_qdrant import Qdrant
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document as LCDocument
import time
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

if not QDRANT_URL or not QDRANT_API_KEY:
    raise ValueError("QDRANT_URL and QDRANT_API_KEY must be set in environment variables")

qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
search = DuckDuckGoSearchRun()


def detect_user_language(query: str) -> str:
    """Detects if query is in English or Hinglish based on content."""
    if not query or not query.strip():
        return "en"
    try:
        lang = detect(query)
        hinglish_indicators = {"kaise", "kya", "hai", "kar", "ho", "nahi", "chahiye", "ya", "aur", "ke", "mein", "se",
                               "ko", "ki", "ka"}
        if lang == "hi" or any(word in query.lower().split() for word in hinglish_indicators):
            return "hi"
        return "en"
    except Exception as e:
        logger.warning(f"Language detection failed: {e}. Defaulting to English.")
        return "en"


def validate_file_path(file_path: str, lang: str = "en") -> bool:
    """Validates if the file exists and is readable."""
    if os.path.exists(file_path) and os.access(file_path, os.R_OK):
        return True
    error_msg = "File not found or inaccessible" if lang == "en" else "File nahi mila ya access nahi hai"
    raise FileNotFoundError(f"❌ {error_msg}: {file_path}. Please check file path.")


def get_embedding_model():
    """Returns the embedding model for vector search."""
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


def combine_results(vector_results: list, keyword_results: list, vector_weight: float = 0.6,
                    keyword_weight: float = 0.4) -> list:
    """Combines vector and keyword search results with weighted scoring."""
    combined = {}
    for doc in vector_results:
        combined[doc.page_content] = combined.get(doc.page_content, 0) + vector_weight
    for doc in keyword_results:
        combined[doc.page_content] = combined.get(doc.page_content, 0) + keyword_weight
    sorted_results = sorted(combined.items(), key=lambda x: x[1], reverse=True)
    return [text for text, _ in sorted_results[:3]]


@tool
def sim_swap_split(file_path: str = '../RAG_Files/simcard_manager.txt',
                   collection_name: str = "simcard_data", query: str = ""):
    """
    Loads a text file, splits it, embeds content into Qdrant, and performs hybrid RAG (vector + keyword search).
    Falls back to web search if no results are found. Responds in English or Hinglish based on query.
    """
    lang = detect_user_language(query)
    validate_file_path(file_path, lang)

    logger.info(f"Loading text file: {file_path}")
    loader = TextLoader(file_path)
    text_documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    documents = text_splitter.split_documents(text_documents)
    if not documents:
        error_msg = "No documents loaded from text file." if lang == "en" else "Text file se koi documents nahi mile."
        raise ValueError(f"⚠️ {error_msg}")

    embedding_model = get_embedding_model()
    qdrant = Qdrant.from_documents(
        documents=documents,
        embedding=embedding_model,
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
        collection_name=collection_name
    )
    logger.info(f"Embeddings stored in Qdrant: {collection_name}")

    bm25_retriever = BM25Retriever.from_documents(documents)
    bm25_retriever.k = 3

    result = {
        "vector_retriever": qdrant,
        "keyword_retriever": bm25_retriever,
        "search_results": [],
        "language": lang
    }

    if query and query.strip():
        try:
            vector_results = qdrant.similarity_search(query, k=3)
            keyword_results = bm25_retriever.get_relevant_documents(query)
            combined_texts = combine_results(vector_results, keyword_results)

            if combined_texts:
                result["search_results"] = combined_texts
            else:
                log_msg = f"No results for '{query}'. Performing web search..." if lang == "en" else f"'{query}' ke liye kuch nahi mila. Web search kar raha hoon..."
                logger.info(log_msg)
                web_query = f"Airtel {query}"
                web_results = search.run(web_query)
                prefix = "Found on web" if lang == "en" else "Web se mila"
                result["search_results"] = [f"{prefix}: {web_results[:500]}..."]
                result["source"] = "web"
        except Exception as e:
            logger.error(f"Search error: {e}")
            error_msg = f"Error processing query: {str(e)}. Try again or contact Airtel support." if lang == "en" else f"Query process karne mein error: {str(e)}. Dobara try karo ya Airtel support se contact karo."
            result["search_results"] = [f"❌ {error_msg}"]

    return result


@tool
def embed_csv_to_qdrant(csv_path: str = '../RAG_Files/airtel_plans.csv',
                        collection_name: str = "airtel_plans", query: str = ""):
    """
    Loads a CSV file, embeds content into Qdrant, and performs hybrid RAG (vector + keyword search).
    Falls back to web search if no results are found. Responds in English or Hinglish based on query.
    """
    lang = detect_user_language(query)
    validate_file_path(csv_path, lang)

    logger.info(f"Loading CSV: {csv_path}")
    loader = CSVLoader(file_path=csv_path)
    documents = loader.load_and_split()
    if not documents:
        error_msg = "No documents loaded from CSV." if lang == "en" else "CSV se koi documents nahi mile."
        raise ValueError(f"⚠️ {error_msg}")

    embedding_model = get_embedding_model()
    qdrant = Qdrant.from_documents(
        documents=documents,
        embedding=embedding_model,
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
        collection_name=collection_name
    )
    logger.info(f"Embeddings stored in Qdrant: {collection_name}")

    bm25_retriever = BM25Retriever.from_documents(documents)
    bm25_retriever.k = 3

    result = {
        "vector_retriever": qdrant,
        "keyword_retriever": bm25_retriever,
        "search_results": [],
        "language": lang
    }

    if query and query.strip():
        try:
            vector_results = qdrant.similarity_search(query, k=3)
            keyword_results = bm25_retriever.get_relevant_documents(query)
            combined_texts = combine_results(vector_results, keyword_results)

            if combined_texts:
                result["search_results"] = combined_texts
            else:
                log_msg = f"No results for '{query}'. Performing web search..." if lang == "en" else f"'{query}' ke liye kuch nahi mila. Web search kar raha hoon..."
                logger.info(log_msg)
                web_query = f"Airtel {query}"
                web_results = search.run(web_query)
                prefix = "Found on web" if lang == "en" else "Web se mila"
                result["search_results"] = [f"{prefix}: {web_results[:500]}..."]
                result["source"] = "web"
        except Exception as e:
            logger.error(f"Search error: {e}")
            error_msg = f"Error processing query: {str(e)}. Try again or contact Airtel support." if lang == "en" else f"Query process karne mein error: {str(e)}. Dobara try karo ya Airtel support se contact karo."
            result["search_results"] = [f"❌ {error_msg}"]

    return result


@tool
def embed_pdf_to_qdrant(pdf_path: str = '../RAG_Files/airtel_num.pdf',
                        collection_name: str = "airtel_numbers", query: str = ""):
    """
    Loads a PDF file, embeds content into Qdrant, and performs hybrid RAG (vector + keyword search).
    Falls back to web search if no results are found. Responds in English or Hinglish based on query.
    """
    lang = detect_user_language(query)
    validate_file_path(pdf_path, lang)

    logger.info(f"Loading PDF: {pdf_path}")
    loader = PyPDFLoader(file_path=pdf_path)
    documents = loader.load_and_split()
    if not documents:
        error_msg = "No documents loaded from PDF." if lang == "en" else "PDF se koi documents nahi mile."
        raise ValueError(f"⚠️ {error_msg}")

    embedding_model = get_embedding_model()
    qdrant = Qdrant.from_documents(
        documents=documents,
        embedding=embedding_model,
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
        collection_name=collection_name
    )
    logger.info(f"Embeddings stored in Qdrant: {collection_name}")

    bm25_retriever = BM25Retriever.from_documents(documents)
    bm25_retriever.k = 3

    result = {
        "vector_retriever": qdrant,
        "keyword_retriever": bm25_retriever,
        "search_results": [],
        "language": lang
    }

    if query and query.strip():
        try:
            vector_results = qdrant.similarity_search(query, k=3)
            keyword_results = bm25_retriever.get_relevant_documents(query)
            combined_texts = combine_results(vector_results, keyword_results)

            if combined_texts:
                result["search_results"] = combined_texts
            else:
                log_msg = f"No results for '{query}'. Performing web search..." if lang == "en" else f"'{query}' ke liye kuch nahi mila. Web search kar raha hoon..."
                logger.info(log_msg)
                web_query = f"Airtel {query}"
                web_results = search.run(web_query)
                prefix = "Found on web" if lang == "en" else "Web se mila"
                result["search_results"] = [f"{prefix}: {web_results[:500]}..."]
                result["source"] = "web"
        except Exception as e:
            logger.error(f"Search error: {e}")
            error_msg = f"Error processing query: {str(e)}. Try again or contact Airtel support." if lang == "en" else f"Query process karne mein error: {str(e)}. Dobara try karo ya Airtel support se contact karo."
            result["search_results"] = [f"❌ {error_msg}"]

    return result


@tool
def embed_faq_pdf(pdf_path: str = '../RAG_Files/FAQ.pdf',
                  collection_name: str = "airtel_faq", query: str = ""):
    """
    Loads an FAQ PDF, embeds content into Qdrant, and performs hybrid RAG (vector + keyword search).
    Falls back to web search if no results are found. Responds in English or Hinglish based on query.
    """
    lang = detect_user_language(query)
    validate_file_path(pdf_path, lang)

    logger.info(f"Loading FAQ PDF: {pdf_path}")
    loader = PyPDFLoader(file_path=pdf_path)
    documents = loader.load_and_split()
    if not documents:
        error_msg = "No documents loaded from FAQ PDF." if lang == "en" else "FAQ PDF se koi documents nahi mile."
        raise ValueError(f"⚠️ {error_msg}")

    embedding_model = get_embedding_model()
    qdrant = Qdrant.from_documents(
        documents=documents,
        embedding=embedding_model,
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
        collection_name=collection_name
    )
    logger.info(f"Embeddings stored in Qdrant: {collection_name}")

    bm25_retriever = BM25Retriever.from_documents(documents)
    bm25_retriever.k = 3

    result = {
        "vector_retriever": qdrant,
        "keyword_retriever": bm25_retriever,
        "search_results": [],
        "language": lang
    }

    if query and query.strip():
        try:
            vector_results = qdrant.similarity_search(query, k=3)
            keyword_results = bm25_retriever.get_relevant_documents(query)
            combined_texts = combine_results(vector_results, keyword_results)

            if combined_texts:
                result["search_results"] = combined_texts
            else:
                log_msg = f"No results for '{query}'. Performing web search..." if lang == "en" else f"'{query}' ke liye kuch nahi mila. Web search kar raha hoon..."
                logger.info(log_msg)
                web_query = f"Airtel {query}"
                web_results = search.run(web_query)
                prefix = "Found on web" if lang == "en" else "Web se mila"
                result["search_results"] = [f"{prefix}: {web_results[:500]}..."]
                result["source"] = "web"
        except Exception as e:
            logger.error(f"Search error: {e}")
            error_msg = f"Error processing query: {str(e)}. Try again or contact Airtel support." if lang == "en" else f"Query process karne mein error: {str(e)}. Dobara try karo ya Airtel support se contact karo."
            result["search_results"] = [f"❌ {error_msg}"]

    return result


@tool
def telecom_news_lookup(query: str = "latest telecom policies India 2025"):
    """
    Fetches telecom news or policy updates using DuckDuckGo search.
    Responds in English or Hinglish based on query language.
    """
    lang = detect_user_language(query)
    log_msg = f"Searching telecom news: {query}" if lang == "en" else f"Telecom news search kar raha hoon: {query}"
    logger.info(log_msg)

    try:
        web_results = search.run(query)
        prefix = "Found on web" if lang == "en" else "Web se mila"
        return {"search_results": [f"{prefix}: {web_results[:500]}..."], "language": lang, "source": "web"}
    except Exception as e:
        logger.error(f"Web search error: {e}")
        error_msg = f"Error fetching news: {str(e)}. Try again later." if lang == "en" else f"News fetch karne mein error: {str(e)}. Baad mein try karo."
        return {"search_results": [f"❌ {error_msg}"], "language": lang}


# Tool Groups
plan_tools = [embed_csv_to_qdrant, search]
sim_tools = [sim_swap_split, search]
num_tools = [embed_pdf_to_qdrant, search]
policy_tools = [telecom_news_lookup, search]
faq_tools = [embed_faq_pdf, search]