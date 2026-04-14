import os
from langchain_community.document_loaders import TextLoader
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import CHROMA_PATH

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

if os.path.exists(CHROMA_PATH):
    print("--- LOADING EXISTING CHROMA DB ---")
    vectorstore = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings,
    )
else:
    print("--- CREATING NEW CHROMA DB ---")
    docs = TextLoader("nadra_info.txt").load()
    chunks = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=150,
    ).split_documents(docs)

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_PATH,
    )

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})