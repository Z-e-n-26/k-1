"""
RAG Summarizer — Streamlit App (MVP)
------------------------------------

Implements an end-to-end workflow per the requirements:
- Customer uploads a document
- Processor (internal) runs AI summarization (RAG-style over chunks)
- QC/QA reviews & approves/edits (can refine via Perplexity Pro)
- System issues invoice and alert to customer
- Customer pays and downloads original + finished document
- Basic security, audit logging, and role-based views

Run locally (Python 3.10+ recommended):

    pip install streamlit python-docx pdfminer.six scikit-learn pandas reportlab requests

    streamlit run app.py

Environment variables (optional):
- STRIPE_SECRET_KEY (optional; if set, Stripe Checkout is used in Billing)

Login (defaults for demo):
- Customer:  user / user123
- Processor: proc / proc123
- Admin:     admin / admin123

Note: This file is a single-file MVP for clarity. For production, split into modules, add
proper auth (e.g., OAuth), object storage (S3/GCS), and real email + payments.
"""
import numpy as np
import os
import re
import uuid
import sqlite3
import hashlib
import logging
import requests
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import re
import streamlit as st
import pandas as pd

# Text extraction
try:
    from pdfminer.high_level import extract_text as pdf_extract_text
except Exception:
    pdf_extract_text = None

try:
    import docx
except Exception:
    docx = None

# Simple RAG / vectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Invoice PDF
try:
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas as rl_canvas
except Exception:
    A4 = None
    rl_canvas = None

APP_DIR = Path(__file__).parent if '__file__' in globals() else Path.cwd()
DATA_DIR = APP_DIR / 'data'
DATA_DIR.mkdir(exist_ok=True)
DB_PATH = APP_DIR / 'app.db'

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("rag_mvp")

# ---------------------- Utilities ----------------------

def sha256(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()

@dataclass
class Job:
    job_id: str
    user_id: str
    status: str  # created -> processing -> ready_for_qc -> approved -> invoiced -> paid -> deliverable_ready
    title: str
    original_filename: str
    original_path: str
    summary_path: Optional[str] = None
    qc_notes: Optional[str] = None
    price_cents: int = 5000
    created_at: str = datetime.utcnow().isoformat()
    updated_at: str = datetime.utcnow().isoformat()

# ---------------------- DB ----------------------

def init_db():
    with sqlite3.connect(DB_PATH) as con:
        cur = con.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
              user_id TEXT PRIMARY KEY,
              username TEXT UNIQUE,
              password_hash TEXT,
              role TEXT
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS jobs (
              job_id TEXT PRIMARY KEY,
              user_id TEXT,
              status TEXT,
              title TEXT,
              original_filename TEXT,
              original_path TEXT,
              summary_path TEXT,
              qc_notes TEXT,
              price_cents INTEGER,
              created_at TEXT,
              updated_at TEXT
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS invoices (
              invoice_id TEXT PRIMARY KEY,
              job_id TEXT,
              amount_cents INTEGER,
              status TEXT,
              created_at TEXT
            )
            """
        )
        con.commit()


def seed_demo_users():
    users = [
        (str(uuid.uuid4()), 'user', sha256('user123'), 'customer'),
        (str(uuid.uuid4()), 'proc', sha256('proc123'), 'processor'),
        (str(uuid.uuid4()), 'admin', sha256('admin123'), 'admin'),
    ]
    with sqlite3.connect(DB_PATH) as con:
        cur = con.cursor()
        for uid, uname, pwh, role in users:
            try:
                cur.execute("INSERT INTO users(user_id, username, password_hash, role) VALUES (?,?,?,?)",
                            (uid, uname, pwh, role))
            except sqlite3.IntegrityError:
                pass
        con.commit()


def get_user(username: str):
    with sqlite3.connect(DB_PATH) as con:
        cur = con.cursor()
        cur.execute("SELECT user_id, username, password_hash, role FROM users WHERE username=?", (username,))
        row = cur.fetchone()
        return row


def create_job(job: Job):
    with sqlite3.connect(DB_PATH) as con:
        cur = con.cursor()
        cur.execute(
            """
            INSERT INTO jobs(job_id, user_id, status, title, original_filename, original_path, summary_path, qc_notes, price_cents, created_at, updated_at)
            VALUES(?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                job.job_id, job.user_id, job.status, job.title, job.original_filename,
                job.original_path, job.summary_path, job.qc_notes, job.price_cents,
                job.created_at, job.updated_at
            )
        )
        con.commit()


def update_job(job_id: str, **fields):
    fields['updated_at'] = datetime.utcnow().isoformat()
    keys = ', '.join([f"{k} = ?" for k in fields.keys()])
    values = list(fields.values()) + [job_id]
    with sqlite3.connect(DB_PATH) as con:
        cur = con.cursor()
        cur.execute(f"UPDATE jobs SET {keys} WHERE job_id = ?", values)
        con.commit()


def fetch_jobs(where: str = "", params: Tuple = ()):
    with sqlite3.connect(DB_PATH) as con:
        con.row_factory = sqlite3.Row
        cur = con.cursor()
        cur.execute(f"SELECT * FROM jobs {where}", params)
        rows = cur.fetchall()
        return [dict(row) for row in rows]


def create_invoice(job_id: str, amount_cents: int) -> str:
    invoice_id = str(uuid.uuid4())
    with sqlite3.connect(DB_PATH) as con:
        cur = con.cursor()
        cur.execute(
            "INSERT INTO invoices(invoice_id, job_id, amount_cents, status, created_at) VALUES(?,?,?,?,?)",
            (invoice_id, job_id, amount_cents, 'unpaid', datetime.utcnow().isoformat())
        )
        con.commit()
    return invoice_id


def update_invoice(invoice_id: str, **fields):
    keys = ', '.join([f"{k} = ?" for k in fields.keys()])
    values = list(fields.values()) + [invoice_id]
    with sqlite3.connect(DB_PATH) as con:
        cur = con.cursor()
        cur.execute(f"UPDATE invoices SET {keys} WHERE invoice_id = ?", values)
        con.commit()


def get_invoice(job_id: str):
    with sqlite3.connect(DB_PATH) as con:
        con.row_factory = sqlite3.Row
        cur = con.cursor()
        cur.execute("SELECT * FROM invoices WHERE job_id = ?", (job_id,))
        row = cur.fetchone()
        return dict(row) if row else None

# ---------------------- Text & RAG ----------------------

ALLOWED_EXTS = {'.txt', '.pdf', '.docx'}

def extract_text_from_file(path: Path) -> str:
    ext = path.suffix.lower()
    if ext == '.txt':
        return path.read_text(encoding='utf-8', errors='ignore')
    elif ext == '.pdf':
        if not pdf_extract_text:
            raise RuntimeError("pdfminer.six not installed")
        return pdf_extract_text(str(path))
    elif ext == '.docx':
        if not docx:
            raise RuntimeError("python-docx not installed")
        d = docx.Document(str(path))
        return "\n".join([p.text for p in d.paragraphs])
    else:
        raise ValueError(f"Unsupported file type: {ext}")

def chunk_text(text: str, chunk_size: int = 800, overlap: int = 100) -> List[str]:
    # Simple word-based chunking
    words = re.split(r"\s+", text)
    chunks = []
    i = 0
    while i < len(words):
        chunk = words[i:i+chunk_size]
        chunks.append(' '.join(chunk))
        i += chunk_size - overlap
    return [c.strip() for c in chunks if c.strip()]

def build_vector_index(chunks: List[str]):
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(chunks)
    return vectorizer, X

def retrieve(chunks: List[str], vectorizer, X, query: str, k: int = 5) -> List[Tuple[int, float]]:
    q = vectorizer.transform([query])
    sims = cosine_similarity(q, X)[0]
    ranked = sorted(list(enumerate(sims)), key=lambda x: x[1], reverse=True)
    return ranked[:k]

def simple_extractive_summary(text: str, max_sentences: int = 12) -> str:
    sents = re.split(r"(?<=[.!?])\s+", text)
    if len(sents) <= max_sentences:
        return text.strip()
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(sents)
    centroid = np.asarray(X.mean(axis=0))   # ensure np.ndarray, not np.matrix
    sims = cosine_similarity(X, centroid)
    order = sorted(range(len(sents)), key=lambda i: sims[i,0], reverse=True)[:max_sentences]
    order_sorted = sorted(order)
    return ' '.join([sents[i].strip() for i in order_sorted])

def rag_summarize(full_text: str, instruction: str = "Summarise the document faithfully without omitting core components.") -> str:
    chunks = chunk_text(full_text)
    if not chunks:
        return ""
    vectorizer, X = build_vector_index(chunks)
    # Use a retrieval step guided by prompt – we simulate with a few focused queries
    guiding_queries = [
        "main objectives and scope",
        "critical steps and workflow",
        "inputs, outputs, and actors",
        "security and compliance",
        "billing, invoicing, notifications"
    ]
    selected = set()
    for q in guiding_queries:
        for idx, _ in retrieve(chunks, vectorizer, X, q, k=3):
            selected.add(idx)
    retrieved_text = "\n\n".join(chunks[i] for i in sorted(selected))
    # Final extractive summary over retrieved content (MVP)
    summary = simple_extractive_summary(retrieved_text, max_sentences=14)
    return summary.strip()

# ---------------------- Invoice Generation ----------------------

def generate_invoice_pdf(invoice_id: str, job: Dict, amount_cents: int, out_path: Path):
    if rl_canvas is None or A4 is None:
        # Fallback: write a simple text invoice
        out_path.write_text(
            f"Invoice: {invoice_id}\nJob: {job['job_id']}\nAmount: {amount_cents/100:.2f}\nDate: {datetime.utcnow().isoformat()}\n",
            encoding='utf-8'
        )
        return
    c = rl_canvas.Canvas(str(out_path), pagesize=A4)
    width, height = A4
    y = height - 50
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, y, "INVOICE")
    y -= 30
    c.setFont("Helvetica", 12)
    c.drawString(50, y, f"Invoice ID: {invoice_id}")
    y -= 20
    c.drawString(50, y, f"Job ID: {job['job_id']}")
    y -= 20
    c.drawString(50, y, f"Title: {job['title']}")
    y -= 20
    c.drawString(50, y, f"Amount: ${amount_cents/100:.2f}")
    y -= 20
    c.drawString(50, y, f"Date: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")
    y -= 40
    c.drawString(50, y, "Thank you for your business.")
    c.showPage()
    c.save()

# ---------------------- Security helpers ----------------------

def ensure_safe_filename(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]", "_", name)

def save_uploaded_file(upl, job_id: str) -> Path:
    folder = DATA_DIR / job_id
    folder.mkdir(parents=True, exist_ok=True)
    safe = ensure_safe_filename(upl.name)
    path = folder / safe
    with open(path, 'wb') as f:
        f.write(upl.getbuffer())
    return path

# ---------------------- Auth ----------------------

def login_block():
    st.sidebar.header("Login")
    username = st.sidebar.text_input("Username")
    password = st.sidebar.text_input("Password", type="password")
    if st.sidebar.button("Sign in", use_container_width=True):
        row = get_user(username)
        if not row:
            st.sidebar.error("Unknown user")
            return None
        user_id, uname, pwh, role = row
        if sha256(password) != pwh:
            st.sidebar.error("Incorrect password")
            return None
        st.session_state.user = {"user_id": user_id, "username": uname, "role": role}
        st.rerun()
    return None

# ---------------------- Perplexity Integration ----------------------

PPLX_DEFAULT_MODEL = "sonar-reasoning"  # reasonable default for summarization/refinement

def call_perplexity(prompt: str, context: str, model: str, temperature: float = 0.7, max_tokens: int = 800) -> str:
    """
    Calls Perplexity's Chat Completions API using the API key stored in session state.
    Returns the refined text or an error string prefixed by '❌'.
    """
    api_key = st.session_state.get("perplexity_api_key")
    if not api_key:
        return "⚠️ No Perplexity API key set. Please paste it in the Perplexity Settings expander."

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a helpful summarization and refinement assistant. Keep output faithful to the provided context and avoid fabrications."},
            {"role": "user", "content": f"Instruction:\n{prompt}\n\nContext to refine:\n{context}"}
        ],
        "temperature": float(temperature),
        "max_tokens": int(max_tokens),
    }
    try:
        resp = requests.post("https://api.perplexity.ai/chat/completions", headers=headers, json=payload, timeout=60)
        if not resp.ok:
            # Return helpful error details (commonly 400 when model name or payload invalid)
            return f"❌ Perplexity API error: {resp.status_code} {resp.reason} — {resp.text}"
        data = resp.json()
        # Prefer the standardized OpenAI-like field if present
        if data.get("choices"):
            choice = data["choices"][0]
            if "message" in choice and "content" in choice["message"]:
                return choice["message"]["content"]
            if "text" in choice:  # some providers use 'text'
                return choice["text"]
        return f"❌ Perplexity API: unexpected response format — {data}"
    except Exception as e:
        return f"❌ Perplexity API error: {e}"

# ---------------------- UI Pages ----------------------

def page_customer():
    st.title("📄 Upload & Track")
    st.write("Upload a document to receive an AI summary after QC.")

    # Upload new document
    with st.expander("New submission", expanded=True):
        title = st.text_input("Title", placeholder="e.g., Contract #42")
        upl = st.file_uploader(
            "Original document (.pdf, .docx, .txt)",
            type=["pdf", "docx", "txt"]
        )
        if st.button("Submit", disabled=not (title and upl)):
            job_id = str(uuid.uuid4())
            path = save_uploaded_file(upl, job_id)
            job = Job(
                job_id=job_id,
                user_id=st.session_state.user['user_id'],
                status='created',
                title=title,
                original_filename=upl.name,
                original_path=str(path),
            )
            create_job(job)
            st.success(f"Submitted. Job ID: {job_id}")
            st.rerun()

    # Show jobs for this customer (only ID, title, status)
    st.subheader("My Jobs")
    jobs = fetch_jobs(
        "WHERE user_id = ? ORDER BY created_at DESC",
        (st.session_state.user['user_id'],)
    )
    if jobs:
        df = pd.DataFrame(
            [(j['job_id'], j['title'], j['status']) for j in jobs],
            columns=["Job ID", "Title", "Status"]
        )
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.info("No jobs yet.")

    # Payment + Download
    st.subheader("Payments & Downloads")
    job_id = st.text_input("Enter Job ID to proceed")
    if job_id:
        job = next((j for j in jobs if j['job_id'] == job_id), None)
        if not job:
            st.error("Invalid Job ID.")
            return

        inv = get_invoice(job_id)

        if job["status"] in ("invoiced", "approved"):
            st.warning("Payment required before downloads.")

            with st.form("payment_form"):
                name = st.text_input("Your Name")
                email = st.text_input("Your Email")
                amount = st.number_input(
                    "Enter Payment Amount (USD)",
                    min_value=0.0,
                    value=job["price_cents"] / 100,
                    step=1.0,
                )
                pay = st.form_submit_button("Pay Now")

            if pay:
                if not name or not email or amount <= 0:
                    st.error("Please fill all details correctly.")
                else:
                    # Simulate payment success
                    if inv:
                        update_invoice(inv["invoice_id"], status="paid")
                    update_job(job_id, status="paid")
                    st.success("✅ Payment successful. Invoice marked as paid.")

                    # Generate invoice PDF
                    out_pdf = DATA_DIR / job_id / f"invoice_{inv['invoice_id']}.pdf"
                    generate_invoice_pdf(inv["invoice_id"], job, int(amount * 100), out_pdf)

                    st.success("Invoice generated and ready to download.")

        elif job["status"] in ("paid", "deliverable_ready"):
            st.success("✅ Payment complete. Downloads available:")

            # Original file
            if job["original_path"] and Path(job["original_path"]).exists():
                with open(job["original_path"], "rb") as f:
                    st.download_button(
                        "⬇️ Download Original File",
                        f,
                        file_name=Path(job["original_path"]).name,
                    )

            # Summary file
            if job["summary_path"] and Path(job["summary_path"]).exists():
                with open(job["summary_path"], "rb") as f:
                    st.download_button(
                        "⬇️ Download Summary",
                        f,
                        file_name=Path(job["summary_path"]).name,
                    )

            # Invoice file
            inv = get_invoice(job_id)
            if inv:
                out_pdf = DATA_DIR / job_id / f"invoice_{inv['invoice_id']}.pdf"
                if out_pdf.exists():
                    with open(out_pdf, "rb") as f:
                        st.download_button(
                            "⬇️ Download Invoice PDF",
                            f,
                            file_name=out_pdf.name,
                        )

def page_processor():
    st.title("⚙️ Processor — Summarize & QC Handoff")
    pending = fetch_jobs("WHERE status IN ('created','processing','ready_for_qc') ORDER BY created_at ASC")
    if not pending:
        st.info("No pending jobs.")
        return

    job_ids = [j['job_id'] for j in pending]
    selected = st.selectbox("Pick a job", job_ids)
    job = next(j for j in pending if j['job_id'] == selected)

    st.write(f"**Title:** {job['title']}")
    st.write(f"**Status:** {job['status']}")

    if st.button("Run Summarization (RAG)"):
        update_job(job['job_id'], status='processing')
        try:
            text = extract_text_from_file(Path(job['original_path']))
            summary = rag_summarize(text)
            out_path = DATA_DIR / job['job_id'] / 'summary.txt'
            out_path.write_text(summary, encoding='utf-8')
            update_job(job['job_id'], summary_path=str(out_path), status='ready_for_qc')
            st.success("Summary generated and sent to QC")
        except Exception as e:
            st.error(f"Failed: {e}")

    st.caption("Tip: You can re-run summarization after editing the source or changing logic.")


def strip_markdown(text: str) -> str:
    # Remove bold/italic
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)   # **bold**
    text = re.sub(r"\*(.*?)\*", r"\1", text)       # *italic*
    # Remove headers (##, ### etc.)
    text = re.sub(r"^#+\s*", "", text, flags=re.MULTILINE)
    # Remove horizontal rules (--- or ***)
    text = re.sub(r"^[-*_]{3,}\s*$", "", text, flags=re.MULTILINE)
    # Remove citations like [1], [2][3], [4]
    text = re.sub(r"\[\d+(?:\]\[\d+)*\]", "", text)
    return text.strip()


def page_qc():
    st.title("🧪 QC / QA — Review & Approve")

    # Perplexity settings
    with st.expander("⚙️ Perplexity Settings", expanded=False):
        api_key_input = st.text_input(
            "Paste your Perplexity API Key",
            type="password",
            value=st.session_state.get("perplexity_api_key", "")
        )
        model_choice = st.selectbox(
            "Choose Perplexity model",
            ["sonar-pro", "sonar-reasoning", "sonar-deep-research"],
            index=0
        )
        temperature = st.slider("Temperature", 0.0, 1.5, 0.7, 0.1)
        if st.button("Save Settings"):
            st.session_state["perplexity_api_key"] = api_key_input
            st.session_state["perplexity_model"] = model_choice
            st.session_state["perplexity_temperature"] = float(temperature)
            st.success("Perplexity settings saved for this session.")

    # Fetch jobs
    rows = fetch_jobs("WHERE status = 'ready_for_qc' ORDER BY updated_at ASC")
    if not rows:
        st.info("Nothing waiting for QC.")
        return

    selected = st.selectbox("Pick a job to review", [r['job_id'] for r in rows])
    job = next(r for r in rows if r['job_id'] == selected)

    st.write(f"**Title:** {job['title']}")
    if job['summary_path'] and Path(job['summary_path']).exists():
        current = Path(job['summary_path']).read_text(encoding='utf-8')
    else:
        current = ""
    edited = st.text_area("Editable Summary", value=current, height=300)
    qc_notes = st.text_area("QC Notes (internal)", value=job.get('qc_notes') or "")

    # Refinement section
    st.subheader("Ask Perplexity for Refinement")
    custom_prompt = st.text_input(
        "Custom instruction",
        placeholder="e.g., The result is not satisfied, give me more details and structure it with headings."
    )
    use_model = st.session_state.get("perplexity_model", PPLX_DEFAULT_MODEL)
    use_temp = float(st.session_state.get("perplexity_temperature", 0.7))

    if st.button("Refine with Perplexity"):
        refined = call_perplexity(
            prompt=custom_prompt or "Refine and expand the summary, preserving factual fidelity and adding missing details.",
            context=edited or current,
            model=use_model,
            temperature=use_temp
        )
        if refined.startswith("❌") or refined.startswith("⚠️"):
            st.error(refined)
        else:
            # Clean Markdown + citations
            refined_clean = strip_markdown(refined)

            # Save cleaned text
            Path(job['summary_path']).write_text(refined_clean, encoding='utf-8')
            st.success("Summary refined with Perplexity.")

            # Show refined clean version
            st.text_area("Refined Summary (from Perplexity)", value=refined_clean, height=300, disabled=True)

            # Editable box
            edited = st.text_area("Editable Summary (clean text)", value=refined_clean, height=300)

            # Live preview
            st.markdown("### Preview")
            st.write(edited)

    # Approve / Request changes
    col1, col2 = st.columns(2)
    if col1.button("Approve & Invoice"):
        Path(job['summary_path']).write_text(edited, encoding='utf-8')
        update_job(job['job_id'], qc_notes=qc_notes, status='approved')
        inv_id = create_invoice(job['job_id'], job['price_cents'])
        update_job(job['job_id'], status='invoiced')
        st.success(f"Approved. Invoice {inv_id} created.")
        st.rerun()
    if col2.button("Request Changes"):
        update_job(job['job_id'], qc_notes=qc_notes, status='created')
        st.warning("Sent back to processor.")
        st.rerun()


def page_billing():
    st.title("💳 Billing & Alerts")
    invoiced = fetch_jobs("WHERE status = 'invoiced' ORDER BY updated_at DESC")
    if not invoiced:
        st.info("No invoices pending.")
        return

    selected = st.selectbox("Pick an invoiced job", [j['job_id'] for j in invoiced])
    job = next(j for j in invoiced if j['job_id'] == selected)
    inv = get_invoice(job['job_id'])
    st.write(f"Invoice ID: {inv['invoice_id']}")
    st.write(f"Amount: ${inv['amount_cents']/100:.2f}")

    # Generate invoice PDF
    out_pdf = DATA_DIR / job['job_id'] / f"invoice_{inv['invoice_id']}.pdf"
    if st.button("Generate Invoice PDF"):
        generate_invoice_pdf(inv['invoice_id'], job, inv['amount_cents'], out_pdf)
        st.success("Invoice generated.")

    if out_pdf.exists():
        with open(out_pdf, 'rb') as f:
            st.download_button("Download Invoice PDF", f, file_name=out_pdf.name)

    # Send Alert (simulated)
    default_email = f"{job['job_id'][:8]}@example.com"
    email_to = st.text_input("Customer email", value=default_email)
    if st.button("Send Alert w/ Invoice (Simulated)"):
        # In production, integrate SMTP/provider
        message = f"Subject: Your invoice {inv['invoice_id']}\n\nPlease pay ${inv['amount_cents']/100:.2f} to download your documents."
        (DATA_DIR / job['job_id'] / 'email.txt').write_text(message, encoding='utf-8')
        st.success("Alert saved (simulated email).")

    # Payment (Demo or Stripe)
    stripe_key = os.getenv('STRIPE_SECRET_KEY')
    if stripe_key:
        st.info("Stripe key detected. Implement Checkout session here in production.")
    if st.button("Mark as Paid (Demo)"):
        update_invoice(inv['invoice_id'], status='paid')
        update_job(job['job_id'], status='paid')
        st.success("Payment recorded.")


def page_delivery():
    st.title("📦 Delivery Finalization")
    paid = fetch_jobs("WHERE status = 'paid' ORDER BY updated_at ASC")
    if not paid:
        st.info("No paid jobs.")
        return

    selected = st.selectbox("Pick a paid job", [j['job_id'] for j in paid])
    job = next(j for j in paid if j['job_id'] == selected)

    st.write("Finalize deliverables for customer download.")
    if st.button("Mark deliverable ready"):
        update_job(job['job_id'], status='deliverable_ready')
        st.success("Marked as ready for customer downloads.")


def page_admin():
    st.title("🔒 Admin & Audit")
    st.write("Basic audit and user management.")

    all_jobs = fetch_jobs("ORDER BY created_at DESC")
    if all_jobs:
        st.dataframe(pd.DataFrame(all_jobs), use_container_width=True)

    st.subheader("Add user")
    uname = st.text_input("Username")
    pwd = st.text_input("Password", type='password')
    role = st.selectbox("Role", ["customer","processor","admin"])
    if st.button("Create user"):
        try:
            with sqlite3.connect(DB_PATH) as con:
                cur = con.cursor()
                cur.execute("INSERT INTO users(user_id, username, password_hash, role) VALUES(?,?,?,?)",
                            (str(uuid.uuid4()), uname, sha256(pwd), role))
                con.commit()
            st.success("User created.")
        except sqlite3.IntegrityError:
            st.error("Username already exists.")

# ---------------------- App ----------------------

def main():
    st.set_page_config(page_title="RAG Summarizer", page_icon="🧠", layout="wide")
    init_db()
    seed_demo_users()

    if 'user' not in st.session_state:
        user = login_block()
        st.stop()

    u = st.session_state.user
    st.sidebar.success(f"Signed in as {u['username']} ({u['role']})")
    choice = st.sidebar.radio("Navigate", [
        "Customer", "Processor", "QC/QA", "Billing", "Delivery", "Admin"
    ])

    if choice == "Customer" and u['role'] in ("customer","admin"):
        page_customer()
    elif choice == "Processor" and u['role'] in ("processor","admin"):
        page_processor()
    elif choice == "QC/QA" and u['role'] in ("processor","admin"):
        page_qc()
    elif choice == "Billing" and u['role'] in ("admin","processor"):
        page_billing()
    elif choice == "Delivery" and u['role'] in ("admin","processor"):
        page_delivery()
    elif choice == "Admin" and u['role'] == "admin":
        page_admin()
    else:
        st.error("You don't have access to this section.")

if __name__ == '__main__':
    main()
# api key: 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"