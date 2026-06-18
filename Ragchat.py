from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from transformers import pipeline

app = FastAPI()

# Load models once
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

llm = pipeline(
    "text-generation",
    model="distilgpt2"
)

# Global variables
chunks = []
index = None

# -----------------------------
# Home Route
# -----------------------------
@app.get("/")
def home():
    return {"message": "PDF RAG Chatbot API Running"}

# -----------------------------
# Upload PDF
# -----------------------------
@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    global chunks, index

    pdf_reader = PdfReader(file.file)

    text = ""

    for page in pdf_reader.pages:
        text += page.extract_text()

    chunks = text.split(".")
    chunks = [c.strip() for c in chunks if c.strip()]

    embeddings = embed_model.encode(chunks)
    embeddings = np.array(embeddings)

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    return {
        "message": "PDF uploaded successfully",
        "chunks": len(chunks)
    }

# -----------------------------
# Question Model
# -----------------------------
class Question(BaseModel):
    question: str

# -----------------------------
# Ask Question
# -----------------------------
@app.post("/ask")
def ask_question(data: Question):
    global chunks, index

    if index is None:
        return {"error": "Upload a PDF first"}

    query_embedding = embed_model.encode([data.question])

    distances, indices = index.search(
        np.array(query_embedding),
        2
    )

    retrieved_text = " ".join(
        [chunks[i] for i in indices[0]]
    )

    prompt = f"""
    Context:
    {retrieved_text}

    Question:
    {data.question}

    Answer:
    """

    response = llm(
        prompt,
        max_length=150
    )

    return {
        "answer": response[0]["generated_text"]
    }