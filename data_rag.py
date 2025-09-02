from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from pydantic import SecretStr
from api import openai_api

# Общие компоненты
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    api_key=SecretStr(openai_api)
)

vector_store = Chroma(
    collection_name="prompt_engineering",
    embedding_function=embeddings,
    persist_directory="./chroma_db"
)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,  # уменьшено для предотвращения превышения лимита токенов
    chunk_overlap=200,
    add_start_index=True
)

def add_document_to_vector_store(content: str, source: str):
    from langchain_core.documents import Document
    doc = Document(page_content=content, metadata={"source": source})
    chunks = text_splitter.split_documents([doc])
    vector_store.add_documents(chunks)
    return len(chunks)
