import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

def load_and_process_document(file_path):
    loader = TextLoader(file_path, encoding='utf-8')
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=200,
        length_function=len
    )
    splits = text_splitter.split_documents(documents)
    return splits

def create_vector_store(splits):
    persist_dir = "./chroma_db"
    embeddings = OpenAIEmbeddings()
    # Si ya existe la base de datos, cargarla; si no, crearla
    if os.path.exists(persist_dir) and os.listdir(persist_dir):
        vectorstore = Chroma(
            persist_directory=persist_dir,
            embedding_function=embeddings
        )
    else:
        vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=embeddings,
            persist_directory=persist_dir
        )
    return vectorstore