# Transportation RAG Chatbot

A simple Python Retrieval-Augmented Generation chatbot for transportation and logistics assistance.

The app retrieves relevant information from local text files in `data/transport_docs` 
and uses that context to answer questions. It can run in two modes:

- Retrieval-only fallback: returns a grounded answer from the best matching documents.
- Optional Ollama generation: sends the retrieved context to a local Ollama model.

## Features

- Streamlit chat interface
- Local transportation knowledge base
- TF-IDF document retrieval with scikit-learn
- Source snippets shown with each answer
- Optional local LLM through Ollama
- Focused prompt for transportation boundaries

## Project Structure

```text
transportation-ai-chatbot/
  app.py
  rag.py
  ollama_client.py
  requirements.txt
  data/
    transport_docs/
      bus_route_101.txt
      metro_blue_line.txt
      shipping_faq.txt
```

## Setup

```powershell
cd transportation-ai-chatbot
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run

```powershell
streamlit run app.py
```

Then open the local URL shown by Streamlit.

## Optional Ollama Mode

Install and run Ollama, then pull a model:

```powershell
ollama pull llama3.1
ollama serve
```

In another terminal:

```powershell
$env:OLLAMA_MODEL="llama3.1"
streamlit run app.py
```

If Ollama is not available, the app automatically uses retrieval-only answers.

## Add Your Own Documents

Add `.txt` files to `data/transport_docs`, then restart the Streamlit app.
Use official schedules, route maps, or logistics guides for the best results.
