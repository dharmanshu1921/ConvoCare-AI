# ConvoCareAI: Multi-Agent Customer Support Chatbot

---

## Project Overview

**ConvoCareAI** is a multi-agent customer support chatbot designed to reduce response times and improve service quality by leveraging advanced AI technologies including Large Language Models (LLMs), Retrieval-Augmented Generation (RAG), and multi-agent workflows. This system automates and enhances customer interactions in the telecom domain, providing accurate and context-aware assistance on queries such as FAQs, SIM activation, troubleshooting, recharge plans, telecom policies, helpline numbers, and Airtel store locations.

---

## Key Features

- **Multi-Agent Workflow:** Orchestrated using LangChain and LangGraph to efficiently route and manage diverse customer queries.
- **Retrieval-Augmented Generation (RAG):** Enables contextually relevant and up-to-date responses by integrating external data sources.
- **Real-Time Data Integration:** Uses APIs like Tavily and Serper for fetching live policy updates and store locator information.
- **Embedding Storage & Search:** Utilizes Qdrant vector database for efficient embedding storage and fast similarity searches.
- **Natural Language Processing:** Employs SpaCy and FastText for preprocessing, language detection, and text understanding.
- **File-based Query Handling:** Integrates PyTesseract OCR to extract information from images/documents.
- **User-Friendly Interface:** Built using Streamlit to provide an interactive chatbot experience.
- **Hybrid Work Model:** Adapted to a mix of remote and on-site collaboration due to office constraints.

---

## Technologies & Tools Used

- **Programming Language:** Python  
- **AI & NLP:** Large Language Models (LLMs), LangChain, LangGraph, HuggingFace Embeddings (all-MiniLM-L6-v2), SpaCy, FastText, PyTesseract  
- **Databases & APIs:** Qdrant Vector Database, Tavily API, Serper API  
- **Framework:** Streamlit for UI  
- **Geospatial Data:** Geopy  

---

## Project Architecture

1. **Input Query Processing:** Language detection and preprocessing using FastText and SpaCy.  
2. **Multi-Agent Orchestration:** Different AI agents handle specific query types using LangChain with LangGraph.  
3. **RAG Component:** Queries first trigger retrieval of relevant documents/data from Airtel datasets (CSV, PDF, TXT) and live APIs.  
4. **Response Generation:** LLMs generate natural language responses based on retrieved context.  
5. **Output:** Interactive and dynamic replies served to users through the Streamlit interface.

---

## Contributions & Learnings

- Developed expertise in multi-agent AI system design and deployment.  
- Successfully integrated large-scale telecom datasets with real-time APIs to provide comprehensive support.  
- Tackled challenges in query routing, language identification, and accurate information retrieval.  
- Balanced simultaneous tasks of data preprocessing, agent development, and UI deployment within a fixed timeline.  
- Collaborated effectively across data science, customer experience, and IT teams in a hybrid work environment.

---

## Academic Relevance

The project was supported by knowledge gained in the following courses:  
- Artificial Intelligence (CSE3705)  
- Generative AI and LLMs (CSE3720)  
- Generative AI Agents – Task Automation with LLM Reasoning (CSE3024)

---

## Future Work

- Expand the chatbot to support additional languages and regional dialects.  
- Integrate voice-based queries and multimodal inputs.  
- Enhance agents with adaptive learning from user feedback.  
- Deploy in a cloud environment for scalable access.

---
