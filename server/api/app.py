import os
import time
import pandas as pd
import pinecone
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
import openai
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

# -----------------------------
# Configuration and Initialization
# -----------------------------
# All API keys and regions are loaded from environment variables.
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_REGION   = os.environ.get("PINECONE_REGION")
OPENAI_API_KEY    = os.environ.get("OPENAI_API_KEY")

pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "embeddings-testing-6"
index = pc.Index(name=index_name)

embedding_model = SentenceTransformer('all-MiniLM-L6-v2')


# -----------------------------
# OpenAI and RAG Pipeline Functions
# -----------------------------
openai.api_key = OPENAI_API_KEY
client = openai.OpenAI(api_key=OPENAI_API_KEY)

def retrieve_context(query, top_n=2):
    time.sleep(20)
    query_embedding = embedding_model.encode(query).tolist()
    query_result = index.query(
        vector=query_embedding,
        top_k=top_n,
        include_metadata=True
    )
    flat_documents = [match['metadata']['text'] for match in query_result['matches']]
    context = "\n".join(flat_documents)
    return context

def generate_response(query, context):
    prompt = f"Context: {context}\n\nQuestion: {query}\n\nAnswer:"
    print("\n--- Generated Prompt ---")
    print(prompt)
    print("------------------------\n")
    response = client.chat.completions.create(
        model="gpt-3.5-turbo-0125",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=256,
        temperature=0.7,
    )
    return response.choices[0].message.content

def rag_pipeline(query, top_n=2):
    context = retrieve_context(query, top_n=top_n)
    response = generate_response(query, context)
    return response

# -----------------------------
# Flask Web App Setup
# -----------------------------
app = Flask(__name__, template_folder="../templates")
CORS(app)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat_endpoint():
    if not request.is_json:
        return jsonify({"error": "Request must be in JSON format"}), 400
    data = request.get_json()
    query = data.get('query', '').strip()
    if not query:
        return jsonify({"error": "Query is required"}), 400
    try:
        answer = rag_pipeline(query)
        return jsonify({"query": query, "response": answer})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
