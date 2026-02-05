import streamlit as st
from dotenv import load_dotenv
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import ChatPromptTemplate

from utils.embeddings import get_embeddings
from data_loader.loader import format_docs
from llm import model

# ----------------------------
# Load environment variables
# ----------------------------
load_dotenv()

# ----------------------------
# Page config
# ----------------------------
st.set_page_config(
    page_title="🩺 Medical Assistant",
    page_icon="🩺",
    layout="centered"
)

# ----------------------------
# Custom CSS
# ----------------------------
st.markdown(
    """
    <style>
    body {
        background-color: white;
    }

    .chat-row {
        display: flex;
        gap: 10px;
        align-items: flex-start;
        margin-bottom: 12px;
    }

    .emoji {
        font-size: 28px;
        line-height: 1.2;
    }

    .user-box {
        background-color: #0b5d1e;  /* dark green */
        color: white;
        padding: 12px 16px;
        border-radius: 14px;
        max-width: 80%;
        word-wrap: break-word;
    }

    .bot-box {
        background-color: #7A0250;  /* dark pink */
        color: white;
        padding: 12px 16px;
        border-radius: 14px;
        max-width: 80%;
        word-wrap: break-word;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ----------------------------
# Title
# ----------------------------
st.title("🩺 Medical Assistant")
st.caption("Answers are generated only from verified medical documents.")

# ----------------------------
# Session state (UI only)
# ----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "chat_active" not in st.session_state:
    st.session_state.chat_active = True

# ----------------------------
# Display chat history
# ----------------------------
for role, content in st.session_state.messages:
    if role == "user":
        st.markdown(
            f"""
            <div class="chat-row">
                <div class="emoji">🧑</div>
                <div class="user-box">{content}</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f"""
            <div class="chat-row">
                <div class="emoji">🩺</div>
                <div class="bot-box">{content}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

# ----------------------------
# Pinecone retriever
# ----------------------------
embeddings = get_embeddings()

vectorstore = PineconeVectorStore(
    index_name="medical-chatbot",
    embedding=embeddings
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# ----------------------------
# Prompt (STATELESS RAG)
# ----------------------------
prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a helpful medical assistant. "
        "Answer ONLY from the provided context in 2–3 sentences. "
        "If the answer is not present, say 'I don't know.' "
        "Do NOT show reasoning or analysis."
    ),
    ("human", """
Context:
{context}

Question:
{question}
""")
])

# ----------------------------
# Input box
# ----------------------------
question = st.chat_input(
    "Ask a medical question..."
    if st.session_state.chat_active
    else "Chat ended",
    disabled=not st.session_state.chat_active
)

# ----------------------------
# Handle input
# ----------------------------
if question:
    # Exit words
    if question.lower().strip() in ["exit", "stop", "over"]:
        st.session_state.messages.append(
            ("assistant", "👋 Chat ended. Take care and stay healthy!")
        )
        st.session_state.chat_active = False
        st.rerun()

    # Store user message
    st.session_state.messages.append(("user", question))

    with st.spinner("Searching medical knowledge..."):
        docs = retriever.invoke(question)
        context = format_docs(docs)

        response = model.invoke(
            prompt.format_messages(
                context=context,
                question=question
            )
        )

    st.session_state.messages.append(("assistant", response.content))
    st.rerun()

# ----------------------------
# Clear chat
# ----------------------------
if st.button("🧹 Clear Chat"):
    st.session_state.messages = []
    st.session_state.chat_active = True
    st.rerun()
