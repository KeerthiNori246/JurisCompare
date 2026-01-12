# 📜 JurisCompare #
JurisCompare is a full-stack web application that compares two versions of a legal document (PDFs) and highlights clause-level changes using modern NLP and transformer models.

It is designed to help lawyers, compliance teams, and researchers quickly understand what changed, how it changed, and why it matters.

## 🚀 Features ##
- Upload two versions of a legal document (PDF)
- Clause classification using a fine-tuned LegalBERT model
- Semantic clause matching using sentence embeddings
- Detects clause changes: Added, Modified, Removed, Unchanged
- Extracts key legal keywords from Version 2
- Generates a simplified summary of Version 2
- Visual clause-change breakdown (Chart.js)
- Downloadable CSV report of clause-by-clause comparison
- Production-ready deployment with Gunicorn + Docker
- Secure model handling via Hugging Face Hub

## 🧠 System Architecture (High Level) ##
```
Frontend (HTML + JS + Chart.js)
        |
        v
Flask API (Gunicorn)
        |
        +-- PDF Parsing (pdfplumber)
        +-- Keyword Extraction (KeyBERT)
        +-- Sentence Embeddings (MiniLM)
        +-- Clause Classification (LegalBERT)
        +-- Clause Matching (Cosine Similarity)
        +-- Summary Generation (T5)
        |
        v
CSV Output + JSON Response

```

## 🗂️ Project Structure ##
```
JurisCompare/
│
├── app.py                     # Main Flask application
├── templates/
│   └── index.html             # Frontend UI
│
├── uploads/                   # Temporary runtime uploads
│   └── .gitkeep
│
├── requirements.txt           # Python dependencies
├── Dockerfile                 # Docker build configuration
├── Procfile                   # Deployment process file
│
├── hf_login.py                # Hugging Face login helper
├── upload_model.py            # Upload model to HF Hub
│
├── .gitignore                 # Git ignore rules
├── .gitattributes             # Git LFS config (future models)
│
└── README.md                  # Project documentation
```

## 🧪 Models Used ##
| Task | Model | 
|----------|----------|
| Clause Classification | keerthiN24/legalbert-clause-model  | 
| Sentence Embeddings  | all-MiniLM-L6-v2  | 
| Keyword Extraction  | KeyBERT  | 
| Summarization  | t5-small | 

📌 Note:
The LegalBERT clause classifier is hosted on Hugging Face Hub and not committed to GitHub.

## ⚙️ Installation & Setup (Local) ##
__1️⃣ Clone the Repository__
```
git clone https://github.com/<your-username>/JurisCompare.git
cd JurisCompare
```
__2️⃣ Create a Virtual Environment (Recommended)__
```
python -m venv venv
source venv/bin/activate   # Linux / Mac
venv\Scripts\activate      # Windows
```
__3️⃣ Install Dependencies__
```
pip install -r requirements.txt
```
__4️⃣ Login to Hugging Face (Required Once)__

You must authenticate to download the LegalBERT model:
```
python hf_login.py
```
Paste your Hugging Face access token when prompted.

__5️⃣ Run the Application__
```
python app.py
```
The app will start on:
```
http://localhost:10000
```

## 📊 Output Details ##
- keywords → Top legal keywords from Version 2
- summary → Simplified bullet summary
- clause_counts → Clause type distribution
- change_stats → Added / Modified / Removed / Unchanged
- comparison_csv → Downloadable CSV link

## CSV Output Columns ##
| Column | Description | 
|----------|----------|
| Clause Type  | Legal category (e.g. Termination, Liability)  |
| Version-1 Clause  | Original clause  | 
| Version-2 Clause  | Updated clause  |
| Status  | Change type  | 

## 🔐 Security & Cleanup ##
- Uploaded PDFs are:
     - Stored temporarily
     - Automatically deleted after processing
- CSV file:
     - Deleted immediately after download
- No user data is persisted

## 🧠 Key Design Decisions ##
- Semantic matching instead of string diff → handles paraphrasing
- Confidence + importance scoring → selects meaningful clauses
- Numeric change detection → captures legal-critical value updates
- CPU-only inference → runs on low-resource servers


🔗[Deployment Link](https://juriscompare.onrender.com/)
