<div align="center">

# RAG-Based Chatbot with Streamlit

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=flat-square&logo=python&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-0.1%2B-green?style=flat-square&logo=chainlink&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red?style=flat-square&logo=streamlit&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)

**Chat with any document using RAG**

Upload a file. Ask a question. Get grounded, accurate answers — powered by Retrieval-Augmented Generation.

</div>

---

## Overview

This project is a conversational AI assistant that combines document retrieval with large language model generation. Users can either chat freely with an OpenAI-powered assistant or upload documents and query their content directly. The application is built on LangChain's RAG pipeline and served through a Streamlit web interface.

Two modes are available out of the box:

- **General Chat** — Converse with `gpt-4o-mini` without uploading any files.
- **RAG Chat** — Upload documents, index them into a FAISS vector store, and ask questions grounded in their content.

---

## Features

| Feature | Description |
|---|---|
| Dual chat modes | Switch between general AI chat and document-grounded RAG chat |
| Multi-format ingestion | Supports PDF, TXT, CSV, DOCX, Markdown, HTML, and JSON |
| Conversational memory | Chat history persists across turns in both modes |
| Retrieved context viewer | Inspect the exact document chunks used to generate each answer |
| Robust error handling | Graceful fallbacks for missing Poppler, corrupted files, and API errors |
| Debug instrumentation | Live display of document and chunk counts during indexing |

---

## Supported File Formats

| Format | Extension | Notes |
|---|---|---|
| PDF | `.pdf` | Requires Poppler for full parsing; fallback loader available |
| Plain Text | `.txt` | No additional dependencies |
| CSV | `.csv` | Loaded row-by-row as documents |
| Word Document | `.docx` | Requires `docx2txt` |
| Markdown | `.md` | Parsed as plain text |
| HTML | `.html` | Stripped of tags before indexing |
| JSON | `.json` | Loaded via `UnstructuredFileLoader` |

---

## Architecture

```
User Input
    |
    v
+------------------+
|  Streamlit UI    |  <-- sidebar: API key, file upload, reset
+--------+---------+
         |
         | query
         v
+------------------+       +---------------------------+
|  LangChain Chain |  <--  |  FAISS Vector Store       |
|  (RetrievalQA)   |       |  (document embeddings)    |
+--------+---------+       +---------------------------+
         |                          ^
         | prompt + context         | index on upload
         v                         |
+------------------+       +---------------------------+
|  OpenAI LLM     |       |  Document Loaders          |
|  gpt-4o-mini    |       |  + Text Splitter           |
+------------------+       +---------------------------+
         |
         v
    Answer + Sources
```

---

## Installation

**1. Clone the repository**

```bash
git clone https://github.com/LakshmiSravyaVedantham/RAG-Based-Chatbot-with-Streamlit.git
cd RAG-Based-Chatbot-with-Streamlit
```

**2. Create and activate a virtual environment**

```bash
python -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows
```

**3. Install Python dependencies**

```bash
pip install streamlit langchain langchain-openai langchain-community \
    langchain-text-splitters langchainhub faiss-cpu openai \
    pypdf docx2txt unstructured
pip install "unstructured[pdf]"
```

**4. Install Poppler (required for PDF support)**

| Platform | Command |
|---|---|
| macOS | `brew install poppler` |
| Ubuntu / Debian | `sudo apt update && sudo apt install poppler-utils` |
| Windows | Download from [poppler-windows](https://github.com/oschwartz10612/poppler-windows/releases), extract, and add `bin\` to PATH |

Verify the installation: `pdftotext -v`

---

## Usage

**Start the application**

```bash
streamlit run chatbot.py
```

Open `http://localhost:8501` in your browser.

**General Chat mode**

1. Enter your OpenAI API key in the sidebar.
2. Type any question in the chat input and press Enter.
3. The assistant responds using conversational history from the current session.

**RAG Chat mode**

1. Enter your OpenAI API key in the sidebar.
2. Upload one or more files using the file uploader.
3. Click "Index Uploaded Files" to process and embed the documents.
4. Ask questions about the document content.
5. Expand "Retrieved Context" beneath any answer to inspect the source chunks.

**Reset**

Click "Reset Chat History" in the sidebar to clear all messages and indexed data.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Interface | Streamlit |
| Orchestration | LangChain (RetrievalQA, ConversationChain) |
| Embeddings | OpenAI `text-embedding-ada-002` |
| Vector Store | FAISS (in-memory) |
| LLM | OpenAI `gpt-4o-mini` |
| Document Parsing | PyPDF, docx2txt, Unstructured |
| Text Splitting | LangChain RecursiveCharacterTextSplitter |

---

## Troubleshooting

**Answers say "I don't know"**
- Confirm the file was indexed — chunk and document counts appear in the UI after indexing.
- Use specific keywords from the document rather than paraphrased questions.
- Open the "Retrieved Context" expander to verify relevant chunks are being returned.
- If chunks are too small, increase `chunk_size` in `chatbot.py`.

**Poppler errors on PDF upload**
- Install Poppler and ensure the binary is available on your system PATH.
- Alternatively, convert PDFs to plain text before uploading.

**OpenAI API errors**
- Verify the API key is valid and has remaining quota at [platform.openai.com](https://platform.openai.com/).
- Check for extra whitespace when pasting the key into the sidebar field.

---

## License

This project is licensed under the [MIT License](https://opensource.org/licenses/MIT).

---

## Acknowledgments

Built with [LangChain](https://python.langchain.com/) for RAG orchestration, [Streamlit](https://streamlit.io/) for the web interface, and [OpenAI](https://openai.com/) for language model inference and embeddings.
