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

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_REGION   = os.environ.get("PINECONE_REGION")
OPENAI_API_KEY    = os.environ.get("OPENAI_API_KEY")

pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "embeddings-testing-6"

# Check if the index exists; if not, create it.
existing_indexes = list(pc.list_indexes())
try:
    if index_name not in existing_indexes:
        pc.create_index(
            name=index_name,
            dimension=384,  # Must match the embedding model's dimensionality.
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region=PINECONE_REGION)
        )
        print("Creating Pinecone index... Waiting for it to be ready.")
        time.sleep(20)  # Adjust delay as needed.
except pinecone.openapi_support.exceptions.PineconeApiException as e:
    if "ALREADY_EXISTS" in str(e):
        print("Index already exists. Using the existing index.")
    else:
        raise

# Connect to the Pinecone index (reuse the existing one)
index = pc.Index(name=index_name)

# Initialize Sentence Transformer for embeddings
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# -----------------------------
# Data Processing and Upsert Section
# -----------------------------
def chunk_text(text, max_length=200):
    sentences = text.split('. ')
    chunks = []
    current_chunk = ''
    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= max_length:
            current_chunk += sentence + '. '
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence + '. '
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

file_path = '../data/CST Test Data.xlsx'
df = pd.read_excel(file_path)

vectors_to_upsert = []
for row_index, row in df.iterrows():
    content = f"Product Name: {row['Product Name']}\nProduct Type: {row['Product Type']}\nIssue Description: {row['Issue Description']}\nResolution Suggestion: {row['Resolution Suggestion']}"
    chunks = chunk_text(content)
    for chunk_index, chunk in enumerate(chunks):
        embedding = embedding_model.encode(chunk).tolist()
        vectors_to_upsert.append(
            (f"{row_index}_{chunk_index}", embedding, {"text": chunk, "resource_id": row_index})
        )

batch_size = 100
for i in range(0, len(vectors_to_upsert), batch_size):
    index.upsert(vectors=vectors_to_upsert[i:i + batch_size])

print("✅ Embeddings have been successfully stored in Pinecone.")

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
