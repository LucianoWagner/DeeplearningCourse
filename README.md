# Deeplearning Course - AI Agents & RAG

Este repositorio contiene ejemplos prácticos y scripts desarrollados durante el curso de aprendizaje sobre Agentes de IA y sistemas RAG (Retrieval-Augmented Generation). El proyecto utiliza **LangChain**, **LangGraph**, y la API de **Groq** para inferencia rápida con modelos Llama 3.

## 📂 Estructura del Proyecto

El repositorio incluye varios scripts independientes que demuestran diferentes conceptos:

- **`agent.py`**: Implementación de un agente autónomo utilizando **LangGraph**.
  - Utiliza el patrón ReAct.
  - Tiene acceso a herramientas como **Wikipedia** y una herramienta personalizada para obtener la fecha actual.
  - Muestra el flujo de razonamiento paso a paso.

- **`rag_simulation_script.py`**: Simulación completa de un flujo RAG "end-to-end".
  - Crea una base de conocimiento simulada.
  - Genera embeddings locales usando **FastEmbed** (sin dependencia pesada de PyTorch).
  - Almacena vectores en **ChromaDB**.
  - Realiza recuperación y generación de respuestas usando Groq.

- **`rag_evaluation.py`**: Sistema de evaluación automática para RAG.
  - **Generación de Test**: Usa un LLM "Profesor" para crear preguntas y respuestas basadas en documentos.
  - **Evaluación**: Usa un LLM "Juez" para calificar las respuestas del sistema RAG comparándolas con la respuesta ideal (Ground Truth).

- **`rag_minimal.py`**: Una versión minimalista y condensada de un sistema RAG en menos de 30 líneas de código, ideal para entender los conceptos básicos sin ruido.

## 🛠️ Requisitos Previos

1.  **Python 3.10+**
2.  Una API Key de [Groq](https://console.groq.com/).

## 🚀 Instalación

1.  Clona este repositorio.
2.  Instala las dependencias necesarias:

```bash
pip install langchain langchain-groq langchain-community langgraph fastembed chromadb python-dotenv wikipedia rank_bm25
```

3.  Configura tus variables de entorno. Crea un archivo `.env` en la raíz del proyecto y añade tu API Key de Groq:

```env
GROQ_API_KEY=gsk_tu_api_key_aqui
```

> **Nota**: El archivo `.env` está excluido de git por seguridad.

## ▶️ Uso

### Ejecutar el Agente
```bash
python agent.py
```
Verás cómo el agente decide usar Wikipedia o su herramienta de fecha según la pregunta.

### Ejecutar Simulación RAG
```bash
python rag_simulation_script.py
```
Este script creará una base de datos vectorial temporal y responderá una consulta simulada sobre productos.

### Ejecutar Evaluación RAG
```bash
python rag_evaluation.py
```
Generará un examen automático para el modelo y te mostrará la calificación del juez sobre las respuestas.
