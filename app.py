import streamlit as st
from dotenv import load_dotenv
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import ChatPromptTemplate

from utils.embeddings import get_embeddings
from data_loader.loader import format_docs
from llm import model

# ----------------------------
# Load env
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

    .user-box {
        background-color: #0b5d1e;
        color: white;
        padding: 12px 16px;
        border-radius: 12px;
        margin-bottom: 10px;
        width: fit-content;
        max-width: 80%;
    }

    .bot-box {
        background-color: #7A0250;
        color: white;
        padding: 12px 16px;
        border-radius: 12px;
        margin-bottom: 10px;
        width: fit-content;
        max-width: 80%;
    }

    .chat-row {
        display: flex;
        gap: 10px;
        align-items: flex-start;
    }

    .emoji {
        font-size: 28px;
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
# UI memory (NOT model memory)
# ----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

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
# Prompt (stateless)
# ----------------------------
prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a helpful medical assistant. "
        "Answer ONLY from the provided context in 2 or 3 sentences. "
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
# Input
# ----------------------------
question = st.chat_input("Ask a medical question...")

if question:
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
    st.rerun()
