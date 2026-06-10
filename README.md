# 📄 Multi PDF AI ChatBot using Gemini, LangChain & FAISS

## 📌 Overview

The Multi PDF AI ChatBot is a Retrieval-Augmented Generation (RAG) application that enables users to upload multiple PDF documents and interact with them through natural language conversations.

The system leverages Google's Gemini LLM, LangChain, FAISS Vector Database, and HuggingFace Embeddings to provide accurate, context-aware responses based on the uploaded documents.

Users can ask questions about the content of multiple PDFs simultaneously, making it an efficient solution for document analysis, research, and knowledge retrieval.

---

## 🚀 Features

* Upload and process multiple PDF documents
* Extract text from PDFs automatically
* Intelligent document chunking
* Semantic search using FAISS Vector Store
* Context-aware question answering
* Google Gemini integration for response generation
* Conversational chat interface
* Conversation history tracking
* Streamlit-based interactive UI
* Retrieval-Augmented Generation (RAG) architecture

---

## 🎯 Problem Statement

Organizations and individuals often need to analyze large collections of PDF documents.

Manually searching through multiple files is time-consuming and inefficient.

This project provides an AI-powered assistant capable of:

* Understanding multiple PDF documents
* Retrieving relevant information instantly
* Answering user queries accurately
* Improving productivity and document accessibility

---

## 🏗 System Architecture

User Question
↓
Streamlit Interface
↓
LangChain Pipeline
↓
FAISS Vector Search
↓
Relevant Document Chunks
↓
Google Gemini LLM
↓
AI Generated Response

---

## 📚 Workflow

### Step 1: PDF Upload

Users upload one or more PDF documents through the Streamlit interface.

### Step 2: Text Extraction

The application extracts text using PyPDF2.

### Step 3: Text Chunking

Documents are divided into smaller chunks using RecursiveCharacterTextSplitter.

### Step 4: Embedding Generation

HuggingFace Sentence Transformers generate vector embeddings for each chunk.

### Step 5: Vector Storage

Embeddings are stored in a FAISS vector database for efficient similarity search.

### Step 6: Retrieval

Relevant chunks are retrieved based on semantic similarity to the user's query.

### Step 7: Response Generation

Google Gemini generates accurate responses using the retrieved context.

---

## 🧠 Technologies Used

### Generative AI

* Google Gemini
* Retrieval-Augmented Generation (RAG)

### LLM Framework

* LangChain

### Vector Database

* FAISS

### Embedding Models

* HuggingFace Embeddings
* Sentence Transformers

### Backend

* Python

### Frontend

* Streamlit

### Document Processing

* PyPDF2
* RecursiveCharacterTextSplitter

---

## 📊 Key Capabilities

* Multi-document question answering
* Context-aware retrieval
* Semantic search
* Document intelligence
* AI-powered research assistant
* Knowledge extraction from PDFs

---

## 🔒 Future Enhancements

* Support for DOCX and TXT files
* Persistent Vector Database
* User Authentication
* Chat Memory Across Sessions
* Source Citation Display
* Multi-language Document Support
* Cloud Deployment on AWS/Azure

---

## 👨‍💻 Author

Siddhant Patil

GitHub: https://github.com/siddhantpatilai-tech

Project Repository:
https://github.com/siddhantpatilai-tech/Multi_PDF_ChatBot

Interested in Machine Learning, Deep Learning, Generative AI, RAG Systems, and LLM Applications.
