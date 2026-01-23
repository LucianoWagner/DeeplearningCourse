import os
import re
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langgraph.prebuilt import create_react_agent

# ---------------------------
# 0) Config
# ---------------------------
load_dotenv()

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0
)

# Un LLM aparte para la tool de resumen (evita que el agente se “enrede”)
llm_summarizer = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0
)


# ---------------------------
# 1) Tools
# ---------------------------

@tool
def summarize_text(text: str) -> str:
    """Resume el texto provisto por el usuario en español, de forma clara y breve."""
    prompt = (
        "Resumí el siguiente texto en español, claro y breve. "
        "Usá 5-8 bullets como máximo y cerrá con 1 frase de conclusión.\n\n"
        f"TEXTO:\n{text}"
    )
    resp = llm_summarizer.invoke(prompt)
    return resp.content


# Reemplazo de la tool manual por la tool oficial de LangChain para Wikipedia
api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=1000, lang="es")
wiki_tool = WikipediaQueryRun(api_wrapper=api_wrapper)
wiki_tool.name = "wiki_search"
wiki_tool.description = "Útil para buscar en Wikipedia hechos históricos, biografías, o información enciclopédica. Devuelve un resumen."

tools = [summarize_text, wiki_tool]


# ---------------------------
# 2) Prompt ReAct (Como System Message)
# ---------------------------

few_shot_examples = """
Ejemplo 1:
Usuario: ¿Cuál es la altura del Monte Everest?
Thought: Necesito encontrar la altura del Monte Everest.
Call: wiki_search("Altura Monte Everest")
Result: El Monte Everest tiene una altitud de 8848 metros.
Thought: Tengo el dato.
Respuesta: La altura del Monte Everest es de 8848 metros.

Ejemplo 2:
Usuario: Resumime este texto: "La IA..."
Thought: Debo usar la herramienta de resumen.
Call: summarize_text("La IA...")
Result: - Resumen punto 1...
Respuesta: Aquí tenés el resumen: ...
"""

system_prompt = f"""Sos un agente experto que responde preguntas usando razonamiento y herramientas.
Seguí este proceso:
1. Razoná sobre qué necesitás.
2. Usá herramientas si hace falta (wiki_search para info general, summarize_text para resúmenes).
3. Respondé en Español.

No expliques errores de formato ni menciones pasos internos.
Respondé directamente con la respuesta final para el usuario.


Aquí tenés ejemplos de cómo actuar:
{few_shot_examples}
"""


# ---------------------------
# 3) Crear Agente (LangGraph)
# ---------------------------
# Usamos langgraph.prebuilt.create_react_agent que es la forma moderna
graph = create_react_agent(
    model=llm, 
    tools=tools, 
    prompt=system_prompt  # En esta versión el argumento se llama 'prompt'
)


# ---------------------------
# 4) Pruebas
# ---------------------------
def run_query(q: str):
    print(f"\nPregunta: {q}")
    print("-" * 20)
    inputs = {"messages": [HumanMessage(content=q)]}
    # stream_mode="values" devuelve el estado completo en cada paso
    final_output = None
    for chunk in graph.stream(inputs, stream_mode="values"):
        message = chunk["messages"][-1]
        print(f"[{message.type}]: {message.content[:200]}...") # Imprimimos solo el inicio para no saturar
        if message.type == "ai" and not message.tool_calls:
             final_output = message.content
    
    print(f"\n>> Respuesta Final:\n{final_output}")


if __name__ == "__main__":
    print("\n--- TEST 1: Resumen ---")
    texto = "ReAct combina razonamiento y acción para resolver tareas complejas en LLMs."
    run_query(f"Resumime esto: {texto}")

    print("\n--- TEST 2: Wikipedia ---")
    run_query("¿Quién es el actual presidente de Francia?")


