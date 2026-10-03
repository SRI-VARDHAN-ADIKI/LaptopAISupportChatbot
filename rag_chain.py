import os
from dotenv import load_dotenv

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate

load_dotenv()

DB_PATH = "vector_db"
MODEL_NAME = os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b")

llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name=MODEL_NAME,
    temperature=0.3
)

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
