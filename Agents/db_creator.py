import os
import bs4
from langchain_ollama import OllamaEmbeddings
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma


def create_vectore_store():
    embedding = OllamaEmbeddings(model="mxbai-embed-large:latest", base_url= "http://localhost:11434")
    db_location = "./MCP_db"
    if not os.path.exists(db_location):
        loader = WebBaseLoader("https://www.descope.com/learn/post/mcp")
        data = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        splits = text_splitter.split_documents(data)
        vector_store = Chroma(
            collection_name="MCP_details",
            persist_directory=db_location,
            embedding_function=embedding,
        )
        vector_store.add_documents(documents=splits)
    else:
        vector_store = Chroma(
            collection_name="MCP_details",
            persist_directory=db_location,
            embedding_function=embedding
        )

    retriever = vector_store.as_retriever()
    return retriever