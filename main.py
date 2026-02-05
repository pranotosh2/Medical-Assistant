from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import ChatPromptTemplate
from utils.embeddings import get_embeddings
from data_loader.loader import format_docs
from llm import model
from dotenv import load_dotenv

load_dotenv()



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
# Prompt
# ----------------------------
prompt = ChatPromptTemplate([
    ("system", "You are a helpful medical assistant. Answer only from the context."),
    ("human", """
Context:
{context}

Question:
{question}
""")
])


print("🩺 Medical Chatbot started! Type 'exit' to quit.\n")

while True:
    user_input = input("You: ")

    if user_input.lower() == "exit":
        print("Goodbye!")
        break

    try:
        # 🔹 1. Retrieve docs using ONLY the latest question
        docs = retriever.invoke(user_input)
        context = format_docs(docs)

        # 🔹 3. Call LLM with history + context
        response = model.invoke(
            prompt.format_messages(
                context=context,
                question=user_input
            )
        )

        print(f"Bot: {response.content}\n")

    except Exception as e:
        print(f"Error: {e}\n")
