
# 🩺 Medical Assistant (RAG-based Chatbot)

🚀 **Live Demo:**  
👉 https://medical-assistant-w9bt2juw8kghmos5pwg6qy.streamlit.app/

A professional medical question-answering chatbot built using **LangChain**, **Pinecone**, **Hugging Face embeddings**, and **Streamlit**.  
The assistant answers **only from verified medical documents** using **Retrieval-Augmented Generation (RAG)**.

---

## ✨ Features

- 🩺 Medical-only responses from trusted documents  
- 🔍 Semantic search using Pinecone Vector DB  
- 🧠 HuggingFace MiniLM embeddings  
- 💬 Clean Streamlit chat interface   
- 🚪 Exit commands supported (`exit`, `stop`, `over`)  
- 🧹 Clear chat option  

---

## 🏗️ Architecture

PDF Documents  
↓  
Cleaning  
↓  
Chunking  
↓  
Embeddings (MiniLM)  
↓  
Pinecone Vector DB  
↓  
Retriever  
↓  
LLM  
↓  
Streamlit UI  

---
---

## ▶️ Run Locally

```bash
git clone https://github.com/pranotosh2/Medical-Assistant.git
cd Medical-Assistant
pip install -r requirements.txt
python ingest.py 
streamlit run app.py
```

---

## 🔐 Environment Variables

Create a `.env` file:

```
PINECONE_API_KEY="your_pinecone_key"
HUGGINGFACEHUB_API_TOKEN="your_huggingface_token"
```

---

## 🛡️ Medical Disclaimer

This application is for **educational purposes only**.  
It is **not a substitute for professional medical advice, diagnosis, or treatment**.

---

## 👨‍💻 Author

**Pranotosh Mandal**  
GitHub: https://github.com/pranotosh2
