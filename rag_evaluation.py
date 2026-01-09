import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain.chains import RetrievalQA
from langchain_core.documents import Document
from langchain.evaluation.qa import QAGenerateChain, QAEvalChain

# Cargar API Key
load_dotenv()

# =================================================================
# 1. PREPARAR EL SISTEMA A EVALUAR (Tu RAG)
# =================================================================
print("--- 1. Montando el RAG ---")

# Datos falsos para el ejemplo
docs = [
    Document(page_content="La tarjeta Ualá tiene costo de mantenimiento cero. Es gratis de por vida."),
    Document(page_content="Para pedir la tarjeta tenés que ser mayor de 13 años y tener DNI argentino."),
    Document(page_content="Las inversiones en el fondo común se rescatan en el acto, 24/7."),
]

# Vector Store (Usamos FastEmbed que es local y rápido)
vectorstore = Chroma.from_documents(docs, FastEmbedEmbeddings())

# El LLM que "Rinde el examen" (El alumno)
llm_alumno = ChatGroq(model="llama-3.1-8b-instant", temperature=0)

# La cadena RAG que vamos a evaluar
qa_chain = RetrievalQA.from_chain_type(
    llm=llm_alumno,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(),
    verbose=False 
)

# =================================================================
# 2. GENERACIÓN AUTOMÁTICA DE TEST (Crear el Examen)
# =================================================================
print("--- 2. Generando preguntas de prueba (Test Set) ---")

# Usamos un modelo más potente para crear las preguntas (El Profesor)
llm_profesor = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

# Cadena especializada en inventar preguntas y respuestas
example_gen_chain = QAGenerateChain.from_llm(llm_profesor)

# Generamos 1 par de Pregunta/Respuesta por cada documento
# apply_and_parse lee el doc y devuelve un diccionario {query: ..., answer: ...}
# apply_and_parse lee el doc y devuelve un diccionario {query: ..., answer: ...}
# A veces devuelve una estructura anidada {'qa_pairs': ...}, así que aplanamos si es necesario
raw_examples = example_gen_chain.apply_and_parse([{"doc": d} for d in docs])
examples = [e.get("qa_pairs", e) for e in raw_examples]

print(f"✅ Se generaron {len(examples)} ejemplos de prueba automáticamente.")
print(f"Ejemplo 1: {examples[0]}")

# =================================================================
# 3. CORRIENDO EL EXAMEN (Predicciones)
# =================================================================
print("\n--- 3. El RAG está respondiendo las preguntas... ---")

# Aquí le pedimos a nuestro RAG (qa_chain) que responda las preguntas generadas
# Usamos batch en lugar de apply (que está deprecado)
predictions = qa_chain.batch(examples)

# =================================================================
# 4. EVALUACIÓN (El Juez corrige)
# =================================================================
print("\n--- 4. Evaluación (LLM-as-a-Judge) ---")

# Usamos QAEvalChain para comparar semánticamente
eval_chain = QAEvalChain.from_llm(llm_profesor)

# El juez compara la respuesta real vs la predicción
graded_outputs = eval_chain.evaluate(examples, predictions)

# MOSTRAR RESULTADOS
print("\nRESULTADOS FINALES:")
print("="*60)
for i, example in enumerate(examples):
    print(f"Pregunta: {example['query']}")
    print(f"Realidad (Ground Truth): {example['answer']}")
    print(f"Predicción (Tu Bot):     {predictions[i]['result']}")
    
    # Aquí está la magia: El LLM dice si es CORRECTO o INCORRECTO
    grade = graded_outputs[i]['results'] # En versiones nuevas devuelve 'results'
    print(f"🧑‍⚖️ JUEZ DICE: {grade}") 
    print("-" * 60)