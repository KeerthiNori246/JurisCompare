import os, re, io, csv, nltk, torch, pdfplumber, unicodedata
import pandas as pd
import numpy as np

from flask import Flask, render_template, request, jsonify, send_file

from keybert import KeyBERT
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from difflib import SequenceMatcher
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

# =========================
# CONFIG
# =========================
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
LEGAL_MODEL_PATH = "./legalbert_clause_model"

TOP_N_SUMMARY = 20
TOP_N_KEYWORDS = 80
FINAL_KEYWORD_COUNT = 30
TOP_N_CLAUSES = 25
STOP_WORDS = "english"

FILLER_WORDS = {
    "able","etc","like","said","such","many","various","among","including",
    "hereinafter","thereof","therein","whereas","aforementioned","pursuant",
    "notwithstanding","hereto","thereby","hereby","as aforesaid","for the purposes of",
    "in the event of","from time to time","in accordance with","in relation to","in connection with"
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
nltk.download("punkt")

# =========================
# MODELS (LOAD ONCE)
# =========================
kw_model = KeyBERT(model=EMBEDDING_MODEL)
embed_model = SentenceTransformer(EMBEDDING_MODEL)

summarizer = pipeline(
    "summarization", model="t5-small",
    device=0 if torch.cuda.is_available() else -1
)

tokenizer = AutoTokenizer.from_pretrained(LEGAL_MODEL_PATH)
clause_model = AutoModelForSequenceClassification.from_pretrained(LEGAL_MODEL_PATH).to(device)


# =========================
# PDF EXTRACTION
# =========================
def extract_text_from_pdf(file_path):
    text = ""
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            text += (page.extract_text() or "") + "\n"
    return text


# =========================
# BASIC CLEANING
# =========================
def clean_text(text):
    text = text.lower()
    text = re.sub(r"\bpage\s*\d+\b", "", text)
    text = re.sub(r"_+", " ", text)
    text = re.sub(r"\d+\s+", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def normalize_text(text):
    return unicodedata.normalize("NFKC", text)


def split_sentences(text):
    sents = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip(" .,-•–") for s in sents if len(s.split()) >= 3]


# =========================
# KEYWORD PROCESSING
# =========================
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
        if p in used:
            continue
        keep.append((p, keywords[i][1]))
        sims = cosine_similarity([embeddings[i]], embeddings)[0]
        for j, sim in enumerate(sims):
            if sim > threshold:
                used.add(phrases[j])
    return keep


# =========================
# SUMMARY GENERATION
# =========================
def summarize_text(full_text):
    sentences = split_sentences(full_text)
    if not sentences:
        return [], []

    embeddings = embed_model.encode(sentences)
    sim_matrix = cosine_similarity(embeddings)
    scores = sim_matrix.mean(axis=1)

    top_idx = np.argsort(scores)[-TOP_N_SUMMARY * 2:][::-1]
    top_idx = sorted(top_idx)

    def fmt(s):
        s = s.strip()
        s = re.sub(r'^\d+[\.\)]?\s*', '', s)
        s = re.sub(r'^[•\-–]+\s*', '', s)
        if s and not s[0].isupper():
            s = s[0].upper() + s[1:]
        if not s.endswith(('.', '?', '!')):
            s += '.'
        return re.sub(r'\s+', ' ', s)

    formatted = [fmt(sentences[i]) for i in top_idx]

    unique = []
    seen = set()
    for s in formatted:
        if s.lower() not in seen:
            unique.append(s)
            seen.add(s.lower())

    unique = unique[:TOP_N_SUMMARY]

    simplified = []
    for s in unique:
        out = summarizer("simplify: " + s, max_length=100, min_length=15, do_sample=False)
        txt = out[0]['summary_text']
        if txt and not txt[0].isupper():
            txt = txt[0].upper() + txt[1:]
        if not txt.endswith(('.', '?', '!')):
            txt += '.'
        simplified.append(txt)

    return unique, simplified


# =========================
# CLAUSE CLASSIFICATION
# =========================
def classify_clause(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=256)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        out = clause_model(**inputs)
        probs = torch.softmax(out.logits, dim=1)
        conf, id = torch.max(probs, dim=1)
        return clause_model.config.id2label[id.item()], conf.item()


def pick_top_clauses(sentences):
    rows = []
    for sent in sentences:
        label, conf = classify_clause(sent)
        rows.append({"sentence": sent, "label": label, "confidence": conf})

    df = pd.DataFrame(rows)

    embeddings = embed_model.encode(df["sentence"].tolist())
    mean_emb = embeddings.mean(axis=0)

    df["importance"] = cosine_similarity([mean_emb], embeddings)[0]
    df["final_score"] = df["confidence"] * 0.6 + df["importance"] * 0.4

    return df.sort_values("final_score", ascending=False).head(TOP_N_CLAUSES)


# =========================
# CLAUSE COMPARISON
# =========================
def compare_top_clauses(df1, df2):

    v1_sents = df1["sentence"].tolist()
    v2_sents = df2["sentence"].tolist()

    emb1 = embed_model.encode(v1_sents)
    emb2 = embed_model.encode(v2_sents)

    sim_matrix = cosine_similarity(emb1, emb2)

    used_v2 = set()
    out = []

    for idx, row in enumerate(df1.itertuples(index=False)):
        s1 = row.sentence
        label1 = row.label

        j = sim_matrix[idx].argmax()
        best_sim = sim_matrix[idx][j]

        s2 = v2_sents[j] if best_sim > 0.70 else ""

        if j in used_v2:
            s2 = ""
        else:
            used_v2.add(j)

        if s1 and s2:
            ratio = SequenceMatcher(None, s1, s2).ratio()
            status = "Unchanged" if ratio > 0.95 else "Modified"
        elif s1 and not s2:
            status = "Removed"
        elif s2 and not s1:
            status = "Added"
        else:
            continue

        out.append({
            "Clause Type": label1,
            "Version-1 Clause": s1,
            "Version-2 Clause": s2,
            "Status": status
        })

    return pd.DataFrame(out)


# =========================
# VERSION COMPARISON
# =========================
def compare_versions(text1, text2):

    # keywords
    raw_kw = kw_model.extract_keywords(
        text2, keyphrase_ngram_range=(1, 3),
        stop_words=STOP_WORDS, top_n=TOP_N_KEYWORDS
    )
    cleaned = [(clean_keyword(k), s) for k, s in raw_kw if len(k) > 2]
    unique = deduplicate_keywords(cleaned)
    final_kw = [kw for kw, _ in unique[:FINAL_KEYWORD_COUNT]]

    # summary
    _, summary2 = summarize_text(text2)

    # clauses
    df1 = pick_top_clauses(split_sentences(text1))
    df2 = pick_top_clauses(split_sentences(text2))
    comparison_df = compare_top_clauses(df1, df2)
    
    v2_clause_counts = clause_count_by_type(df2)
    change_stats = change_summary(comparison_df)

    return final_kw, summary2, comparison_df, v2_clause_counts, change_stats


def clause_count_by_type(df):
    return df["label"].value_counts().to_dict()

def change_summary(comp_df):
    summary = {
        "Modified": 0,
        "Added": 0,
        "Unchanged": 0,
        "Removed": 0
    }

    for status in comp_df["Status"]:
        summary[status] += 1

    return summary


# =========================
# FLASK APP
# =========================
app = Flask(__name__)


@app.route("/")
def home():
    return render_template("index.html")





# ------------------------
# NEW: DOCUMENT COMPARISON ROUTE
# ------------------------
@app.route("/compare", methods=["POST"])
def compare():

    if "pdf1" not in request.files or "pdf2" not in request.files:
        return jsonify({"error": "Upload two PDFs."})

    f1 = request.files["pdf1"]
    f2 = request.files["pdf2"]

    p1 = os.path.join(UPLOAD_FOLDER, f1.filename)
    p2 = os.path.join(UPLOAD_FOLDER, f2.filename)

    f1.save(p1)
    f2.save(p2)

    text1 = clean_text(extract_text_from_pdf(p1))
    text2 = clean_text(extract_text_from_pdf(p2))

    keywords, summary, comp_df, clause_counts, change_stats = compare_versions(text1, text2)

    comp_path = os.path.join(UPLOAD_FOLDER, "clause_comparison.csv")
    comp_df.to_csv(comp_path, index=False)

    return jsonify({
    "keywords": keywords,
    "summary": summary,
    "comparison_csv": "/download_comparison",
    "clause_counts": clause_counts,
    "change_stats": change_stats
    })



@app.route("/download_comparison")
def download_comp():
    p = os.path.join(UPLOAD_FOLDER, "clause_comparison.csv")
    return send_file(p, mimetype="text/csv", as_attachment=True,
                     download_name="clause_comparison.csv")



if __name__ == "__main__":
    app.run(debug=True)
