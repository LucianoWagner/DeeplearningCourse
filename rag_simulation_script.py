
import os
import sys

# Instalación automática de dependencias si faltan
try:
    import fastembed
    import langchain_community.embeddings.fastembed
    import langchain_chroma
    import rank_bm25
except ImportError:
    print("📦 Instalando dependencias necesarias (fastembed, chromadb, rank_bm25)...")
    os.system("pip install fastembed langchain-chroma rank_bm25")

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA

def main():
    # 0. Cargar variables
    load_dotenv()
    if not os.getenv("GROQ_API_KEY"):
        print("❌ ERROR: No se encontró GROQ_API_KEY en el environment (.env).")
        return

    print("\n🚀 INICIANDO SIMULACIÓN RAG (MODO FASTEMBED - NO TORCH) 🚀\n")

    # 1. Base de Conocimiento (Documents)
    print("--- PASO 1: CREANDO DATOS ---")
    docs = [
        Document(page_content="Producto: Laptop Gamer X1. Precio: $1500. Specs: NVIDIA RTX 4060, 16GB RAM. Ideal para jugar AAA.", metadata={"id": 1}),
        Document(page_content="Producto: Cafetera Smart Brew. Precio: $200. Specs: WiFi, App control. Para amantes del café.", metadata={"id": 2}),
        Document(page_content="Producto: Silla Ergonómica Pro. Precio: $350. Specs: Soporte lumbar. Para oficina.", metadata={"id": 3})
    ]
    print(f"✅ {len(docs)} documentos creados.\n")

    # 2. Embeddings & VectorStore
    print("--- PASO 2: EMBEDDINGS (FASTEMBED) ---")
    try:
        # FastEmbed usa ONNX, evita errores de DLL de Torch en Windows
        embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5") 
        
        vectorstore = Chroma.from_documents(documents=docs, embedding=embeddings)
        retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 1})
        print("✅ VectorStore inicializado correctamente.\n")
    except Exception as e:
        print(f"❌ ERROR CRÍTICO EN EMBEDDINGS: {e}")
        return

    # 3. Prompt
    print("--- PASO 3: PROMPT ---")
    template = """
    Eres un vendedor experto. Responde usando el contexto.
    
    CONTEXTO RECUPERADO:
    {context}
    
    PREGUNTA: {question}
    """
    PROMPT = PromptTemplate(template=template, input_variables=["context", "question"])
    print("✅ Prompt definido.\n")

    # 4. Chain
    print("--- PASO 4: CADENA RAG ---")
    llm = ChatGroq(temperature=0, model_name="llama-3.3-70b-versatile", api_key=os.getenv("GROQ_API_KEY"))
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT},
        verbose=False
    )
    print("✅ Cadena lista.\n")

    # 5. Ejecución
    print("--- PASO 5: EJECUCIÓN (PRUEBA SEMÁNTICA) ---")
    query = "che, necesito viciar mucho, tendras alguna cafetera o algo parecido para mantenerme despierto?" # Frase vaga, requiere semántica
    print(f"🔍 PREGUNTA: '{query}'")
    
    try:
        response = qa_chain.invoke({"query": query})
        
        print("\n📄 DOC RECUPERADO (¿Entendió la semántica?):")
        for doc in response['source_documents']:
            print(f"   👉 {doc.page_content}")
            
        print("\n🤖 RESPUESTA GROQ:")
        print(response['result'])
        print("\n✅ FIN DEL DEMO.")
        
    except Exception as e:
        print(f"❌ ERROR EN EJECUCIÓN: {e}")

if __name__ == "__main__":
    main()
