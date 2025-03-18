# RAG Chatbot

## Overview
This project is an AI-powered application that extracts text from PDFs, stores embeddings in PostgreSQL using the `pgvector` extension, and provides a Streamlit-based user interface for interaction and question answering.

## Project Structure
```
P05_DocumentBased_QnA_Application
│   .env
│   app.py
│
├───Data
│
└───utils
    │   embeddings.py
    │   llm.py
    │   pdf_loader.py
    │   Vectordb_Connection.py
    │   __init__.py
```

## Prerequisites
- **Python & pip**: Python (3.11+) and pip installed.
- **Docker**: Installed and running.
- **PostgreSQL Client**: For direct database access via `psql`.
- **Git (Optional)**: For repository cloning.
- **Visual Studio (for manual pgvector build)**: If not using Docker, install Visual Studio with C++ support.

## Installation Guide

### 1. Install Python Dependencies
```bash
pip install -r requirements.txt
```

### 2. Install and Run Docker
Download Docker from the official website, install, and verify:
```bash
docker --version
```

### 3. Set Up PostgreSQL and pgvector with Docker
```bash
docker pull ankane/pgvector
docker run --name pgvector-demo -e POSTGRES_PASSWORD=mysecretpassword -p 5432:5432 -d ankane/pgvector
docker ps
```

### 4. Manual Installation of pgvector (if not using Docker)
```bash
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
set "PGROOT=C:\Program Files\PostgreSQL\17"
cd %TEMP%
git clone --branch v0.8.0 https://github.com/pgvector/pgvector.git
cd pgvector
nmake /F Makefile.win
nmake /F Makefile.win install
```

### 5. Create Database and Enable pgvector
Connect to PostgreSQL and run:
```sql
CREATE DATABASE vector_db;
\c vector_db
CREATE EXTENSION vector;
```

## Project Flow and Working
1. **Text Extraction**: Extracts text from uploaded PDF files.
2. **Embedding Generation**: Converts extracted text into vector embeddings using an AI model.
3. **Embedding Storage**: Stores the embeddings in PostgreSQL with `pgvector`.
4. **User Interaction (Q&A)**: Streamlit UI for users to upload PDFs, ask questions, and get relevant answers.

## How to Start the Application
```bash
streamlit run app.py
```

## How to Use the Application
1. **Upload PDF**: Drag and drop your PDF.
2. **Ask Questions**: Enter a question in the input box.
3. **Receive Answers**: The app retrieves and displays relevant results.

## Contributing
Feel free to fork the repository and create a pull request. Contributions are welcome!

