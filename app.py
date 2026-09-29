import os
import re
import hashlib
from pathlib import Path

# pyrefly: ignore [missing-import]
import numpy as np
import faiss

from dotenv import load_dotenv
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
from PyPDF2 import PdfReader
from groq import Groq


# =========================================================
# 1. LOAD ENVIRONMENT VARIABLES
# =========================================================

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")

if not GROQ_API_KEY:
    raise ValueError(
        "GROQ_API_KEY is missing. Add your Groq API key to .env."
    )

client = Groq(api_key=GROQ_API_KEY)


# =========================================================
# 2. INITIALIZE FLASK
# =========================================================

app = Flask(__name__)

DATA_DIR = Path("__data__")
DATA_DIR.mkdir(parents=True, exist_ok=True)

app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024


# =========================================================
# 3. APPLICATION STATE
# =========================================================

vectorstore = None
chat_history = []
rubric_text = ""


class HumanMessage:
    def __init__(self, content):
        self.content = content

    def __repr__(self):
        return f"HumanMessage(content={self.content!r})"


class AIMessage:
    def __init__(self, content):
        self.content = content

    def __repr__(self):
        return f"AIMessage(content={self.content!r})"


# =========================================================
# 4. LIGHTWEIGHT LOCAL TEXT EMBEDDINGS
# No PyTorch, HuggingFace, or OpenAI required
# =========================================================

EMBEDDING_DIMENSION = 384


def create_embedding(text):
    vector = np.zeros(
        EMBEDDING_DIMENSION,
        dtype=np.float32
    )

    tokens = re.findall(r"\w+", text.lower())

    for token in tokens:
        digest = hashlib.blake2b(
            token.encode("utf-8"),
            digest_size=8
        ).digest()

        index = int.from_bytes(
            digest[:4],
            byteorder="little"
        ) % EMBEDDING_DIMENSION

        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vector[index] += sign

    norm = np.linalg.norm(vector)

    if norm > 0:
        vector /= norm

    return vector


# =========================================================
# 5. EXTRACT TEXT FROM PDF DOCUMENTS
# =========================================================

def get_pdf_text(pdf_docs):
    all_documents = []

    for pdf in pdf_docs:
        if not pdf or not pdf.filename:
            continue

        filename = secure_filename(pdf.filename)

        if not filename.lower().endswith(".pdf"):
            continue

        try:
            reader = PdfReader(pdf)

            if reader.is_encrypted:
                raise ValueError(
                    f"{filename} is encrypted. Upload an unlocked PDF."
                )

            page_texts = []

            for page in reader.pages:
                page_texts.append(page.extract_text() or "")

            extracted_text = "\n".join(page_texts).strip()

            if extracted_text:
                all_documents.append(extracted_text)

                # Save extracted text, not the original uploaded file.
                text_filename = DATA_DIR / f"{filename}.txt"

                text_filename.write_text(
                    extracted_text,
                    encoding="utf-8"
                )

        except Exception as exc:
            raise ValueError(
                f"Could not read {filename}: {exc}"
            ) from exc

    return "\n\n".join(all_documents)


# =========================================================
# 6. SPLIT TEXT INTO SMALLER CHUNKS
# =========================================================

def get_text_chunks(text, chunk_size=1000, chunk_overlap=200):
    if chunk_overlap >= chunk_size:
        raise ValueError(
            "Chunk overlap must be smaller than chunk size."
        )

    chunks = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break

        start = end - chunk_overlap

    return chunks


# =========================================================
# 7. CREATE FAISS VECTOR DATABASE
# =========================================================

class FAISSVectorStore:

    def __init__(self, chunks):
        if not chunks:
            raise ValueError(
                "No readable text was found in the PDF."
            )

        self.chunks = chunks

        embeddings = np.vstack([
            create_embedding(chunk)
            for chunk in chunks
        ]).astype(np.float32)

        self.index = faiss.IndexFlatIP(
            EMBEDDING_DIMENSION
        )

        self.index.add(embeddings)

    def search(self, question, k=4):
        if not self.chunks:
            return []

        query_vector = create_embedding(question)
        query_vector = np.asarray(
            [query_vector],
            dtype=np.float32
        )

        number_to_search = min(k, len(self.chunks))

        scores, indices = self.index.search(
            query_vector,
            number_to_search
        )

        results = []

        for score, index in zip(scores[0], indices[0]):
            if index < 0:
                continue

            results.append({
                "text": self.chunks[index],
                "score": float(score)
            })

        return results


def get_vectorstore(text_chunks):
    return FAISSVectorStore(text_chunks)


# =========================================================
# 8. CALL GROQ MODEL
# =========================================================

def ask_groq(messages, temperature=0):
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=messages,
        temperature=temperature,
        max_tokens=1500
    )

    return response.choices[0].message.content or ""


# =========================================================
# 9. ANSWER QUESTIONS ABOUT UPLOADED PDFS
# =========================================================

def answer_pdf_question(question):
    if vectorstore is None:
        raise ValueError(
            "Please upload and process a PDF first."
        )

    retrieved_chunks = vectorstore.search(
        question,
        k=4
    )

    if not retrieved_chunks:
        raise ValueError(
            "No relevant text was found in the PDF."
        )

    context = "\n\n".join(
        item["text"] for item in retrieved_chunks
    )

    messages = [
        {
            "role": "system",
            "content": (
                "You are Chat Scholar, an academic PDF assistant. "
                "Answer using the supplied document context. "
                "If the answer is not available in the context, "
                "say that the uploaded document does not provide "
                "enough information. Do not invent facts.\n\n"
                "DOCUMENT CONTEXT:\n" + context
            )
        }
    ]

    # Include a limited amount of earlier conversation.
    for message in chat_history[-6:]:
        if isinstance(message, HumanMessage):
            messages.append({
                "role": "user",
                "content": message.content
            })

        elif isinstance(message, AIMessage):
            messages.append({
                "role": "assistant",
                "content": message.content
            })

    messages.append({
        "role": "user",
        "content": question
    })

    answer = ask_groq(messages)

    chat_history.append(HumanMessage(question))
    chat_history.append(AIMessage(answer))

    return answer


# =========================================================
# 10. GRADE ESSAYS USING GROQ
# =========================================================

def grade_essay(essay):
    if not essay or not essay.strip():
        return "Please enter an essay before submitting."

    system_prompt = (
        "You are an academic essay evaluator. "
        "Evaluate the essay using the supplied rubric. "
        "If no custom rubric is provided, assess grammar, "
        "clarity, organization, relevance, and supporting evidence. "
        "Respond in English. Provide constructive feedback, "
        "identify strengths and areas for improvement, and "
        "do not invent grading criteria.\n\n"
        "RUBRIC:\n"
        + (
            rubric_text
            if rubric_text
            else "No custom rubric provided."
        )
    )

    answer = ask_groq([
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": "Please evaluate this essay:\n\n" + essay
        }
    ], temperature=0.2)

    # Preserve the line-break behavior of the original template.
    return answer.replace("\n", "<br>")


# =========================================================
# 11. EXTRACT ESSAY TEXT FROM PDF
# =========================================================

def extract_text_from_pdf(pdf_file):
    reader = PdfReader(pdf_file)

    if reader.is_encrypted:
        raise ValueError(
            "The essay PDF is encrypted. Upload an unlocked PDF."
        )

    text = "\n".join(
        page.extract_text() or ""
        for page in reader.pages
    ).strip()

    return text


# =========================================================
# 12. HOME PAGE
# =========================================================

@app.route("/")
def home():
    return render_template("new_home.html")


# =========================================================
# 13. PROCESS UPLOADED DOCUMENTS
# =========================================================

@app.route("/process", methods=["POST"])
def process_documents():
    global vectorstore, chat_history

    try:
        pdf_docs = request.files.getlist("pdf_docs")

        if not any(
            pdf and pdf.filename
            for pdf in pdf_docs
        ):
            return "Please select at least one PDF.", 400

        raw_text = get_pdf_text(pdf_docs)

        if not raw_text.strip():
            return (
                "No readable text was found. "
                "Scanned PDFs may require OCR.",
                400
            )

        text_chunks = get_text_chunks(raw_text)
        vectorstore = get_vectorstore(text_chunks)
        chat_history = []

        return redirect(url_for("chat"))

    except Exception:
        app.logger.exception("PDF processing failed")

        return (
            "PDF processing failed. Check the Terminal "
            "for the detailed error.",
            500
        )


# =========================================================
# 14. CHAT WITH PDF
# =========================================================

@app.route("/chat", methods=["GET", "POST"])
def chat():
    global chat_history

    if request.method == "POST":
        question = request.form.get(
            "user_question",
            ""
        ).strip()

        if not question:
            return "Please enter a question.", 400

        if vectorstore is None:
            return "Please upload and process a PDF first.", 400

        try:
            answer_pdf_question(question)

        except Exception:
            app.logger.exception("PDF chat failed")

            return (
                "Chat failed. Check your Groq API key, "
                "model name, and Terminal traceback.",
                500
            )

    return render_template(
        "new_chat.html",
        chat_history=chat_history
    )


# =========================================================
# 15. PDF CHAT PAGE
# =========================================================

@app.route("/pdf_chat", methods=["GET", "POST"])
def pdf_chat():
    return render_template("new_pdf_chat.html")


# =========================================================
# 16. ESSAY GRADING PAGE
# =========================================================

@app.route("/essay_grading", methods=["GET", "POST"])
def essay_grading():
    global rubric_text

    result = None
    input_text = ""

    if request.method == "POST":

        submitted_rubric = request.form.get(
            "essay_rubric",
            ""
        ).strip()

        if submitted_rubric:
            rubric_text = submitted_rubric

            return render_template(
                "new_essay_grading.html",
                result=None,
                input_text=""
            )

        try:
            uploaded_file = request.files.get("file")

            if uploaded_file and uploaded_file.filename:
                input_text = extract_text_from_pdf(
                    uploaded_file
                )
            else:
                input_text = request.form.get(
                    "essay_text",
                    ""
                ).strip()

            if not input_text:
                result = (
                    "Please enter an essay or upload "
                    "a readable PDF."
                )
            else:
                result = grade_essay(input_text)

        except Exception:
            app.logger.exception("Essay grading failed")

            result = (
                "Essay grading failed. Check the Terminal "
                "for the detailed error."
            )

    return render_template(
        "new_essay_grading.html",
        result=result,
        input_text=input_text
    )


# =========================================================
# 17. ESSAY RUBRIC PAGE
# =========================================================

@app.route("/essay_rubric", methods=["GET", "POST"])
def essay_rubric():
    return render_template("new_essay_rubric.html")


# =========================================================
# 18. HANDLE COMMON UPLOAD ERRORS
# =========================================================

@app.errorhandler(413)
def file_too_large(error):
    return (
        "The uploaded file is too large. "
        "The maximum request size is 20 MB.",
        413
    )


# =========================================================
# 19. RUN APPLICATION
# =========================================================

if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )