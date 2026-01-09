import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_core.tools import Tool, tool
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent
from datetime import date

# Cargar entorno
load_dotenv()

# 1. Configurar el "Cerebro" (Reasoning Engine)
llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)

# 2. Configurar Herramientas
# A. Wikipedia
wikipedia = WikipediaAPIWrapper()
wiki_tool = Tool(
    name="Wikipedia",
    func=wikipedia.run,
    description="Útil para buscar hechos históricos, biografías o información enciclopédica."
)

# B. Herramienta Personalizada (Custom Tool: Fecha)
@tool
def get_today_date(query: str = "") -> str:
    """Retorna la fecha actual. Ideal para preguntas sobre 'hoy', 'fecha' o 'día'.
    Ignora cualquier input que se le pase."""
    return str(date.today())

tools = [wiki_tool, get_today_date]

# 3. Inicializar el Agente con LangGraph
# create_react_agent crea un grafo de estado que maneja automáticamente el ciclo de razonamiento
print("Inicializando Agente LangGraph...")
agent_executor = create_react_agent(llm, tools)

# Función helper para ejecutar y mostrar resultados paso a paso
def run_demo(query):
    print(f"\n>>> EJECUTANDO: {query}")
    inputs = {"messages": [HumanMessage(content=query)]}
    
    # stream_mode="values" nos da el estado actualizado del grafo en cada paso
    for chunk in agent_executor.stream(inputs, stream_mode="values"):
        # El último mensaje es el más reciente (puede ser del usuario, del modelo o de una herramienta)
        message = chunk["messages"][-1]
        
        # Imprimimos de forma legible según el tipo de mensaje
        print(f"\n[{message.type.upper()}]:")
        print(message.content)

# --- PRUEBAS ---

# Caso 1: Pregunta que requiere Wikipedia
print("\n--- CASO 1: Wikipedia ---")
run_demo("¿Cuando nació Lionel Messi?")

# Caso 2: Pregunta que requiere Custom Tool
print("\n--- CASO 2: Custom Tool ---")
run_demo("¿Cuál es la fecha de hoy?")