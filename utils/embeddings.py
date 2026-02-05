from langchain_huggingface import HuggingFaceEmbeddings

# ----------------------------
# Embeddings
# ----------------------------

# It maps sentences & paragraphs to a 384 dimensional dense vector space

def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
