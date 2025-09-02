from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_chroma import Chroma
from pydantic import SecretStr
from api import openai_api

# Промпт
prompt = ChatPromptTemplate.from_template(
    """Ответь на вопрос, используя только приведённый ниже контекст.
Если ответа нет в контексте, напиши 'К сожалению нет информации по этому вопросу!'.

В каждом ответе обязательно указывай [Источник], из которого взята информация.
Не придумывай ничего вне контекста. Если ответ частичный — укажи это.
Отвечай кратко и по делу.

[КОНТЕКСТ]
{context}

[ВОПРОС]
{question}

[ОТВЕТ]
"""
)

# LLM
llm = ChatOpenAI(
    model="gpt-4o",
    api_key=SecretStr(openai_api),
    temperature=0.7
)

# Эмбеддинги и векторное хранилище
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    api_key=SecretStr(openai_api)
)

vector_store = Chroma(
    collection_name="prompt_engineering",
    embedding_function=embeddings,
    persist_directory="./chroma_db"
)

def answer_question(question: str) -> str:
    retrieved_docs = vector_store.similarity_search(question, k=3)
    # Отладочный вывод метаданных
    for i, doc in enumerate(retrieved_docs):
        print(f"Фрагмент {i+1} metadata: {doc.metadata}")
    # Формируем контекст с указанием источника
    docs_content = "\n".join([
        f"[Источник: {doc.metadata.get('source', 'неизвестно')}]\n{doc.page_content}" for doc in retrieved_docs
    ])
    message = prompt.invoke({"question": question, "context": docs_content})
    response = llm.invoke(message)
    return str(response.content)
