import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from langchain_core.messages import AIMessage, HumanMessage

from app.sinonimos import cargar_sinonimos, reemplazar_sinonimos
from app.embeddings import load_and_process_document, create_vector_store
from app.qa_chain import setup_qa_chain
from .db import get_db, engine, Base
from .models import Chat, Message
from sqlalchemy.orm import Session

Base.metadata.create_all(bind=engine)

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

os.environ["OPENAI_API_KEY"] = "YOUR_OPENAI_API_KEY"  # Reemplaza con tu clave de API de OpenAI

sinonimos = cargar_sinonimos()
file_path = "conocimiento.txt"
splits = load_and_process_document(file_path)
vectorstore = create_vector_store(splits)
qa_chain = setup_qa_chain(vectorstore)
chat_history = []

@app.route('/qa', methods=['POST'])
def qa():
    data = request.get_json()
    chat_id = data.get("chat_id")
    question = data.get("question", "")

    if not question:
        return jsonify({"error": "No se proporcionó una pregunta"}), 400

    db: Session = next(get_db())
    try:
        if chat_id:
            chat = db.query(Chat).filter(Chat.id == chat_id).first()
        else:
            chat = None

        if not chat:
            # Create a new chat if chat_id is not provided or chat is not found
            new_chat = Chat(name="New Chat")  # You can customize the name
            db.add(new_chat)
            db.commit()
            chat = new_chat
            chat_id = chat.id  # Get the new chat's ID

        chat_history = []
        for msg in chat.messages:
            if msg.type == "human":
                chat_history.append(HumanMessage(content=msg.content))
            elif msg.type == "ai":
                chat_history.append(AIMessage(content=msg.content))

        prompt_normalizado = reemplazar_sinonimos(question, sinonimos)
        response = qa_chain.invoke({
            "question": prompt_normalizado,
            "chat_history": chat_history
        })

        human_message = HumanMessage(content=question)
        ai_message = AIMessage(content=response)

        chat.messages.append(Message(type="human", content=question, chat_id=chat_id))
        chat.messages.append(Message(type="ai", content=response, chat_id=chat_id))
        db.commit()

        chat_history.append(human_message)
        chat_history.append(ai_message)

        chat_history_serializable = [
            {"type": msg.type, "content": msg.content} for msg in chat_history
        ]

        return jsonify({"response": response, "chat": chat_history_serializable})
    finally:
        db.close()
        
@app.route('/particiones', methods=['GET'])
def particiones():
    # Obtener todos los documentos almacenados en la base de datos vectorial
    # Esto depende de la implementación de Chroma y LangChain
    # Usamos vectorstore.get() para obtener los documentos
    docs = []
    if hasattr(vectorstore, "get"):
        # get() retorna un dict con 'documents' como lista de textos
        data = vectorstore.get()
        docs = data.get("documents", [])
    return jsonify({"particiones": docs})