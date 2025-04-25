import os
import json
from flask import Flask, request, jsonify
from flask_cors import CORS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
import re
from dotenv import load_dotenv


app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

def cargar_sinonimos(archivo="filtros.json"):
    try:
        with open(archivo, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Advertencia: No se encontró el archivo {archivo}")
        return {}
    except json.JSONDecodeError:
        print(f"Error: El archivo {archivo} no es un JSON válido")
        return {}

sinonimos = cargar_sinonimos()


def reemplazar_sinonimos(texto):
    if not sinonimos:
        return texto
    
    palabras = texto.lower().split()
    texto_reemplazado = []
    
    for palabra in palabras:
        reemplazada = False
        for palabra_principal, alternativas in sinonimos.items():
            if palabra in alternativas or palabra == palabra_principal:
                texto_reemplazado.append(palabra_principal)
                reemplazada = True
                break
        if not reemplazada:
            texto_reemplazado.append(palabra)
    
    return " ".join(texto_reemplazado)

def create_embeddings(txt_path):
    with open(txt_path, "r", encoding="utf-8") as file:
        text = file.read()
    
    chunks = text.split("\n\n")
   
    embeddings = HuggingFaceEmbeddings(
        model_name=os.getenv("EMBEDDINGS_MODEL", "ibm-granite/granite-embedding-278m-multilingual"))
    knowledge_base = Chroma.from_texts(chunks, embeddings, persist_directory="./chroma_db")
    return knowledge_base

txt_path = "conocimiento.txt"
knowledge_base = create_embeddings(txt_path) if os.path.exists(txt_path) else None

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    prompt = data.get('prompt', '')
    
    if not prompt:
        return jsonify({"error": "Debe enviar un prompt válido."}), 400
    
    # Reemplazar sinónimos en el prompt
    prompt_normalizado = reemplazar_sinonimos(prompt)
    print(f"Prompt original: {prompt}")
    print(f"Prompt normalizado: {prompt_normalizado}")
    
    if not knowledge_base:
        return jsonify({"error": "No se pudo cargar la base de conocimientos."}), 500
    
    docs = knowledge_base.similarity_search(prompt_normalizado, 4)
    print("\nDocumentos relevantes encontrados:")
    for i, doc in enumerate(docs, 1):
        print(f"\nDocumento {i}:")
        print(doc.page_content)
        print("Metadatos:", doc.metadata)
    llm = ChatOpenAI(model_name=os.getenv("LLM_MODEL", "gpt-4o"))
    
    chain = load_qa_chain(llm, chain_type="stuff")
    response = chain.run(input_documents=docs, question=prompt_normalizado)
    response = re.sub(r"\\*", "", response)
    
    return jsonify({"response": response})

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000)
