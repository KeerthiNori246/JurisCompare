import os, re, io, csv, nltk, torch, pdfplumber
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, jsonify, send_file
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

nltk.download('punkt')

# ============================
# CONFIG
# ============================
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
LEGAL_MODEL_PATH = "./legalbert_clause_model"
TOP_N_KEYWORDS = 80
FINAL_KEYWORD_COUNT = 30
TOP_N_SUMMARY = 20
TOP_N_CLAUSES = 25
STOP_WORDS = "english"

FILLER_WORDS = {
    "able", "etc", "like", "said", "such", "many", "various",
    "among", "including", "hereinafter", "thereof", "therein",
    "whereas", "aforementioned", "pursuant", "notwithstanding",
    "hereto", "thereby", "hereby", "as aforesaid", "for the purposes of",
    "in the event of", "from time to time", "in accordance with",
    "in relation to", "in connection with"
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load heavy models once
kw_model = KeyBERT(model=EMBEDDING_MODEL)
embed_model = SentenceTransformer(EMBEDDING_MODEL)
summarizer = pipeline("summarization", model="t5-small", device=0 if torch.cuda.is_available() else -1)
tokenizer = AutoTokenizer.from_pretrained(LEGAL_MODEL_PATH)
clause_model = AutoModelForSequenceClassification.from_pretrained(LEGAL_MODEL_PATH).to(device)

app = Flask(__name__)

# ============================
# UTILITIES
# ============================
def extract_text_from_pdf(file_path):
    text = ""
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            text += (page.extract_text() or "") + "\n"
    return text

def clean_text(text):
    text = text.lower()
    text = re.sub(r"\bpage\s*\d+\b", "", text)
    text = re.sub(r"_+", " ", text)
    text = re.sub(r"\d+\s+", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def split_sentences(text):
    sents = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip(" .,-•–") for s in sents if len(s.split()) >= 3]

def clean_keyword(phrase):
    words = phrase.split()
    seen = []
    for w in words:
        if w not in seen and w not in FILLER_WORDS:
            seen.append(w)
    return " ".join(seen)

def deduplicate_keywords(keywords, threshold=0.75):
    phrases = [kw for kw, _ in keywords]
    embeddings = embed_model.encode(phrases)
    keep, used = [], set()
    for i, p in enumerate(phrases):
        if p in used: continue
        keep.append((p, keywords[i][1]))
        sims = cosine_similarity([embeddings[i]], embeddings)[0]
        for j, sim in enumerate(sims):
            if sim > threshold: used.add(phrases[j])
    return keep

def classify_clause(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=256)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = clause_model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)
        conf, cls_id = torch.max(probs, dim=1)
        return clause_model.config.id2label[cls_id.item()], conf.item()

# ============================
# ROUTES
# ============================

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/analyze", methods=["POST"])
def analyze():
    if "pdf" not in request.files:
        return jsonify({"error": "No file uploaded."}), 400

    file = request.files["pdf"]
    if file.filename == "":
        return jsonify({"error": "No file selected."}), 400

    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    # ====== STEP 1: Extract and preprocess ======
    text = extract_text_from_pdf(file_path)
    cleaned_text = clean_text(text)
    sentences = split_sentences(cleaned_text)

    # ====== STEP 2: Keyword extraction ======
    raw_keywords = kw_model.extract_keywords(
        cleaned_text, keyphrase_ngram_range=(1,3),
        stop_words=STOP_WORDS, top_n=TOP_N_KEYWORDS
    )
    cleaned_keywords = [(clean_keyword(k), s) for k,s in raw_keywords if len(k)>2]
    unique_keywords = deduplicate_keywords(cleaned_keywords)
    final_keywords = unique_keywords[:FINAL_KEYWORD_COUNT]

    # ====== STEP 3: Extractive summary (UPDATED) ======
    sentence_embeddings = embed_model.encode(sentences)
    similarity_matrix = cosine_similarity(sentence_embeddings)
    sentence_scores = similarity_matrix.mean(axis=1)

    top_indices = np.argsort(sentence_scores)[-TOP_N_SUMMARY*2:][::-1]  # pick extra to remove duplicates
    top_indices_sorted = sorted(top_indices)

    def format_sentence(sentence):
        s = sentence.strip()
        s = re.sub(r'^\d+[\.\)]?\s*', '', s)
        s = re.sub(r'\d+\.\w+', '', s)
        s = re.sub(r'^[•\-–]+\s*', '', s)
        if s and not s[0].isupper():
            s = s[0].upper() + s[1:]
        if not s.endswith(('.', '?', '!')):
            s += '.'
        s = re.sub(r'\s+', ' ', s)
        s = s.replace('..', '.')
        return s.strip()

    summary_sentences = [format_sentence(sentences[i]) for i in top_indices_sorted]

    # Remove duplicates while keeping order
    seen = set()
    unique_summary = []
    for s in summary_sentences:
        if s.lower() not in seen:
            unique_summary.append(s)
            seen.add(s.lower())
    summary_sentences = unique_summary[:TOP_N_SUMMARY]

    # Simplify sentences
    simplified_summary = []
    for s in summary_sentences:
        result = summarizer("simplify: " + s, max_length=100, min_length=15, do_sample=False)
        simplified_text = result[0]['summary_text']
        if simplified_text and not simplified_text[0].isupper():
            simplified_text = simplified_text[0].upper() + simplified_text[1:]
        if not simplified_text.endswith(('.', '?', '!')):
            simplified_text += '.'
        simplified_summary.append(simplified_text)

    # ====== STEP 4: Clause classification ======
    clause_results = []
    for sent in sentences:
        label, conf = classify_clause(sent)
        clause_results.append({"sentence": sent, "label": label, "confidence": conf})
    df = pd.DataFrame(clause_results)

    mean_emb = np.mean(embed_model.encode(df["sentence"].tolist()), axis=0)
    df["importance"] = cosine_similarity([mean_emb], embed_model.encode(df["sentence"].tolist()))[0]
    df["final_score"] = df["confidence"]*0.6 + df["importance"]*0.4
    top_clauses = df.sort_values("final_score", ascending=False).head(TOP_N_CLAUSES)

    csv_buf = io.StringIO()
    top_clauses.to_csv(csv_buf, index=False)
    csv_buf.seek(0)

    csv_path = os.path.join(UPLOAD_FOLDER, "top_25_clauses.csv")
    top_clauses.to_csv(csv_path, index=False)

    # ====== RETURN ALL RESULTS ======
    return jsonify({
        "keywords": [kw for kw, _ in final_keywords],
        "summary": simplified_summary,
        "csv_file": "/download_csv"
    })


@app.route("/download_csv")
def download_csv():
    csv_path = os.path.join(UPLOAD_FOLDER, "top_25_clauses.csv")
    return send_file(csv_path, mimetype="text/csv", as_attachment=True, download_name="top_25_clauses.csv")

if __name__ == "__main__":
    app.run(debug=True)
