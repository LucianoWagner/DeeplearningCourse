
import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_groq import ChatGroq
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# Cargar API Key
load_dotenv()

# --- 3 LÍNEAS PARA DATOS Y VECTORES ---
docs = [
    Document(page_content="Laptop Gamer X1. NVIDIA RTX 4060. Ideal para jugar AAA.", metadata={"id": 1}),
    Document(page_content="Cafetera Smart Brew. WiFi. Para amantes del café.", metadata={"id": 2}),
]
# FastEmbed + Chroma en una línea
vectorstore = Chroma.from_documents(docs, FastEmbedEmbeddings())

# --- 3 LÍNEAS PARA EL CEREBRO (CHAIN) ---
prompt = ChatPromptTemplate.from_template("Responde solo con el contexto provisto: {context}. Pregunta: {input}")
llm = ChatGroq(model="llama-3.3-70b-versatile")
rag_chain = create_retrieval_chain(vectorstore.as_retriever(), create_stuff_documents_chain(llm, prompt))

# --- EJECUCIÓN ---
# Quitamos el emoji para evitar error de encoding en tu consola de Windows
print("RESPUESTA:", rag_chain.invoke({"input": "quiero viciar"})["answer"])
