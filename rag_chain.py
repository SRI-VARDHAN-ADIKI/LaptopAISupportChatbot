import os
from dotenv import load_dotenv

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate

try:
    import streamlit as st
except ImportError:  # pragma: no cover
    st = None

load_dotenv()

DB_PATH = "vector_db"
MODEL_NAME = os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b")


def get_groq_api_key():
    api_key = os.getenv("GROQ_API_KEY")
    if api_key:
        return api_key

    if st is not None:
        try:
            secret = st.secrets.get("GROQ_API_KEY")
            if secret:
                return str(secret)
        except Exception:
            pass

    try:
        from dotenv import dotenv_values
        env_values = dotenv_values(".env") or {}
        key = env_values.get("GROQ_API_KEY")
        if key:
            return str(key)
    except Exception:
        pass

    return None


API_KEY = get_groq_api_key()
llm = ChatGroq(
    groq_api_key=API_KEY,
    model_name=MODEL_NAME,
    temperature=0.3
) if API_KEY else None

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

db = Chroma(
    persist_directory=DB_PATH,
    embedding_function=embeddings
)

retriever = db.as_retriever(search_kwargs={"k": 4})

prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are an intelligent IT support assistant.

Guidelines:
- Use provided context if relevant.
- If context is insufficient, use general technical knowledge.
- Blend both naturally when applicable.
- Never mention documentation, databases, or sources.
- Avoid repeating previously suggested steps.
- Be conversational and professional.

Context:
{context}

User Input:
{question}

Helpful Response:
"""
)
def ask_question(question: str):
    if llm is None:
        raise RuntimeError(
            "GROQ_API_KEY is missing. Add it as a Streamlit Cloud secret named GROQ_API_KEY or set it in the environment."
        )

    # New LangChain retriever API
    docs = retriever.invoke(question)

    context_text = "\n\n".join(
        [doc.page_content for doc in docs]
    ) if docs else ""

    response = llm.invoke(
        prompt.format(
            context=context_text,
            question=question
        )
    )

    return {
        "answer": response.content,
        "docs_used": docs
    }
