# ⚖️ Epstein Files RAG MVP

A public, free-to-access AI research assistant for exploring the **Epstein Files** (`teyler/epstein-files-20k` dataset).

This tool allows you to:

- **Ask questions** about the files and get fact-based answers.
- **View Sources**: Every answer is grounded in specific documents, with filenames provided.
- **Chat History**: Ask follow-up questions in a conversational interface.
- **Multiple Models**: Supports **Google Gemini** (Free), **xAI (Grok)**, and **Groq**.

## 🚀 Quick Start

### 1. Prerequisites

- Python 3.10+
- An API Key (One of):
  - **Google Gemini** (Free tier available)
  - **xAI (Grok)**
  - **Groq**

### 2. Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/your-username/epstein-rag.git
cd epstein-rag
pip install -r requirements.txt
```

### 3. Setup Environment

Create a `.env` file in the root directory:

```bash
# Optional: Pre-set your keys here (or enter them in the UI)
GOOGLE_API_KEY=your_key_here
# OPENAI_API_KEY=your_grok_or_groq_key_here
```

### 4. Ingest Data (First Run Only)

Download and index the dataset (this creates the local vector database):

```bash
python ingest.py
```

*Note: This processes the dataset from Hugging Face. It may take a few minutes.*

### 5. Run the App

```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`.

## 🛠 Features

- **Strict No-Speculation Policy**: The AI is instructed to answer *only* from the documents.
- **Source Transparency**: Clickable filenames and text chunks are shown for verification.
- **Rate Limit Handling**: Automatic retries for API constraints.
- **Model Flexibility**: Switch providers instantly in the sidebar.

## ⚠️ Disclaimer

This tool is for research purposes only. It uses large language models which can occasionally hallucinate or make errors. Always verify claims against the provided source documents. The dataset is sourced publicly from `teyler/epstein-files-20k` on Hugging Face.
