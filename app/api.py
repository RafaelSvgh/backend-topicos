import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from langchain_core.messages import AIMessage, HumanMessage

from app.sinonimos import cargar_sinonimos, reemplazar_sinonimos
from app.embeddings import load_and_process_document, create_vector_store
from app.qa_chain import setup_qa_chain

app = Flask(__name__)
CORS(app)

os.environ["OPENAI_API_KEY"] = ""

sinonimos = cargar_sinonimos()
file_path = "conocimiento.txt"
splits = load_and_process_document(file_path)
vectorstore = create_vector_store(splits)
qa_chain = setup_qa_chain(vectorstore)
chat_history = []

@app.route('/qa', methods=['POST'])
def qa():
    data = request.get_json()
    question = data.get("question", "")
    if not question:
        return jsonify({"error": "No se proporcionó una pregunta"}), 400
    prompt_normalizado = reemplazar_sinonimos(question, sinonimos)
    response = qa_chain.invoke({
        "question": prompt_normalizado,
        "chat_history": chat_history
    })
    chat_history.append(HumanMessage(content=question))
    chat_history.append(AIMessage(content=response))
    return jsonify({"response": response})