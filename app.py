"""
AI in Human Resource Management - AI-powered recruitment and employee management
A professional HR SaaS-style Streamlit application powered by Google Gemini.

Run with:
    pip install -r requirements.txt
    streamlit run app.py
"""

import os
import io
import json
import sqlite3
import re
import time
from datetime import date, datetime
from contextlib import contextmanager

import streamlit as st
import pandas as pd
import plotly.express as px
from pypdf import PdfReader

# ----------------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------------

APP_TITLE = "AI in Human Resource Management"
APP_TAGLINE = "AI-powered recruitment and employee management"
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hr_assistant.db")

# The Gemini model used for all AI calls. Change here (or via the sidebar's
# Advanced settings) to switch models without touching the rest of the code.
DEFAULT_GEMINI_MODEL = "gemini-3.5-flash"

JOB_ROLES = [
    "Data Analyst",
    "Business Analyst",
    "Software Engineer",
    "HR Analytics",
    "Finance Analyst",
    "Marketing Analytics",
    "Product Manager",
    "Sales Executive",
    "Operations Manager",
    "Customer Support",
]

STATUS_OPTIONS = ["Active", "On Leave", "Resigned", "Terminated"]

st.set_page_config(
    page_title=f"{APP_TITLE} | {APP_TAGLINE}",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------------------------------
# STYLING
# ----------------------------------------------------------------------------

def inject_css() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

        html, body, [class*="css"]  {
            font-family: 'Inter', sans-serif;
        }

        :root {
            --navy: #0a1628;
            --navy-light: #16273f;
            --accent: #3b82f6;
            --accent-light: #60a5fa;
            --bg: #0b1120;
            --card-bg: #131c2e;
            --card-bg-hover: #182338;
            --border: #263450;
            --text-dark: #f1f5f9;
            --text-muted: #94a3b8;
            --green: #4ade80;
            --amber: #fbbf24;
            --red: #f87171;
        }

        /* Base app background + default text color */
        .stApp {
            background-color: var(--bg);
            color: var(--text-dark);
        }
        body, p, span, label, li, div {
            color: var(--text-dark);
        }
        h1, h2, h3, h4, h5, h6 {
            color: var(--text-dark) !important;
        }

        #MainMenu, footer {visibility: hidden;}

        /* Keep the Streamlit header alive so the sidebar toggle keeps working;
           just hide the deploy/settings toolbar buttons inside it. */
        header[data-testid="stHeader"] {
            background-color: transparent;
            display: flex !important;
            visibility: visible !important;
            opacity: 1 !important;
            height: 3rem !important;
            z-index: 999999 !important;
        }
        div[data-testid="stToolbar"] {
            visibility: hidden;
        }
        button[data-testid="stSidebarCollapseButton"] {
            display: flex !important;
            visibility: visible !important;
            opacity: 1 !important;
            position: relative !important;
            z-index: 9999999 !important;
        }
        button[data-testid="stSidebarCollapsedControl"] {
            display: flex !important;
            visibility: visible !important;
            opacity: 1 !important;
            position: fixed !important;
            top: 0.75rem !important;
            left: 0.75rem !important;
            z-index: 9999999 !important;
        }

        /* Sidebar */
        section[data-testid="stSidebar"] {
            background-color: var(--navy);
            border-right: 1px solid var(--border);
        }
        section[data-testid="stSidebar"] * {
            color: #e2e8f0 !important;
        }
        section[data-testid="stSidebar"] .stRadio > label {
            font-weight: 600;
        }
        section[data-testid="stSidebar"] hr {
            border-color: rgba(255,255,255,0.12);
        }
        section[data-testid="stSidebar"] div[data-baseweb="radio"] label {
            background: var(--navy-light);
            border-radius: 8px;
            padding: 0.4rem 0.6rem;
            margin-bottom: 0.25rem;
        }

        .brand-title {
            font-size: 1.35rem;
            font-weight: 800;
            color: #ffffff !important;
            letter-spacing: -0.02em;
            margin-bottom: 0;
        }
        .brand-sub {
            font-size: 0.78rem;
            color: #94a3b8 !important;
            margin-bottom: 1.2rem;
        }

        .page-header {
            font-size: 1.9rem;
            font-weight: 800;
            color: var(--text-dark) !important;
            letter-spacing: -0.02em;
            margin-bottom: 0.15rem;
        }
        .page-subtext {
            color: var(--text-muted) !important;
            font-size: 0.95rem;
            margin-bottom: 1.6rem;
        }

        .hr-card {
            background: var(--card-bg);
            border: 1px solid var(--border);
            border-radius: 14px;
            padding: 1.3rem 1.5rem;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.25);
            margin-bottom: 1rem;
            color: var(--text-dark);
        }
        .hr-card * { color: var(--text-dark); }

        .metric-card {
            background: var(--card-bg);
            border: 1px solid var(--border);
            border-radius: 14px;
            padding: 1.2rem 1.4rem;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.25);
        }
        .metric-label {
            color: var(--text-muted) !important;
            font-size: 0.82rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.04em;
        }
        .metric-value {
            color: #ffffff !important;
            font-size: 2rem;
            font-weight: 800;
            margin-top: 0.15rem;
        }

        .badge {
            display: inline-block;
            padding: 0.22rem 0.65rem;
            border-radius: 999px;
            font-size: 0.78rem;
            font-weight: 700;
        }
        .badge-strong { background: rgba(74, 222, 128, 0.15); color: var(--green) !important; }
        .badge-good { background: rgba(59, 130, 246, 0.18); color: var(--accent-light) !important; }
        .badge-moderate { background: rgba(251, 191, 36, 0.15); color: var(--amber) !important; }
        .badge-low { background: rgba(248, 113, 113, 0.15); color: var(--red) !important; }

        .rank-1 { border-left: 4px solid #eab308; }
        .rank-2 { border-left: 4px solid #94a3b8; }
        .rank-3 { border-left: 4px solid #b45309; }

        /* Buttons */
        .stButton>button {
            background-color: var(--accent);
            color: white !important;
            border-radius: 8px;
            border: none;
            font-weight: 600;
            padding: 0.5rem 1.2rem;
            transition: background-color 0.15s ease, transform 0.1s ease;
        }
        .stButton>button:hover {
            background-color: var(--accent-light);
            color: white !important;
            transform: translateY(-1px);
        }
        .stButton>button p { color: white !important; }

        /* Quick-action icon tiles on the dashboard */
        .tile-btn button {
            background: var(--card-bg) !important;
            border: 1px solid var(--border) !important;
            color: var(--text-dark) !important;
            border-radius: 14px !important;
            padding: 1.4rem 0.8rem !important;
            width: 100%;
            font-size: 1rem !important;
            font-weight: 700 !important;
            box-shadow: 0 2px 8px rgba(0,0,0,0.25);
        }
        .tile-btn button:hover {
            background: var(--card-bg-hover) !important;
            border-color: var(--accent) !important;
            color: var(--accent-light) !important;
            transform: translateY(-2px);
        }
        .tile-btn button p { color: inherit !important; font-size: 1rem !important; font-weight: 700 !important; }

        div[data-testid="stMetricValue"] {
            color: #ffffff !important;
        }
        div[data-testid="stMetricLabel"] {
            color: var(--text-muted) !important;
        }

        /* Inputs, selects, text areas */
        .stTextInput input, .stTextArea textarea, .stNumberInput input {
            background-color: var(--card-bg) !important;
            color: var(--text-dark) !important;
            border: 1px solid var(--border) !important;
        }
        div[data-baseweb="select"] > div {
            background-color: var(--card-bg) !important;
            color: var(--text-dark) !important;
            border-color: var(--border) !important;
        }
        div[data-baseweb="select"] * { color: var(--text-dark) !important; }
        ul[data-testid="stSelectboxVirtualDropdown"] {
            background-color: var(--card-bg) !important;
        }
        ul[data-testid="stSelectboxVirtualDropdown"] li {
            color: var(--text-dark) !important;
        }
        .stFileUploader section {
            background-color: var(--card-bg) !important;
            border: 1px dashed var(--border) !important;
        }
        .stFileUploader section * { color: var(--text-dark) !important; }
        .stSlider label, .stSlider span { color: var(--text-dark) !important; }

        /* Tabs */
        button[data-baseweb="tab"] {
            color: var(--text-muted) !important;
        }
        button[data-baseweb="tab"][aria-selected="true"] {
            color: var(--accent-light) !important;
        }
        div[data-baseweb="tab-highlight"] {
            background-color: var(--accent) !important;
        }
        div[data-baseweb="tab-border"] {
            background-color: var(--border) !important;
        }

        /* Expander */
        details {
            background-color: var(--card-bg) !important;
            border: 1px solid var(--border) !important;
            border-radius: 10px !important;
        }
        summary { color: var(--text-dark) !important; }

        /* DataFrame / tables */
        .stDataFrame {
            border-radius: 10px;
            overflow: hidden;
            border: 1px solid var(--border);
        }

        /* Alerts (info/success/warning/error) keep readable text on dark bg */
        div[data-testid="stAlert"] p {
            color: var(--text-dark) !important;
        }

        /* Forms */
        div[data-testid="stForm"] {
            background-color: var(--card-bg);
            border: 1px solid var(--border);
            border-radius: 14px;
            padding: 1.2rem;
        }

        /* Hero banner with the app title, shown at the top of the Dashboard */
        .hero-banner {
            background: linear-gradient(120deg, #0a1628 0%, #16273f 55%, #1c3d5c 100%);
            border: 1px solid var(--border);
            border-radius: 18px;
            padding: 1.8rem 2.2rem;
            margin-bottom: 1.6rem;
            box-shadow: 0 4px 18px rgba(0, 0, 0, 0.35);
            position: relative;
            overflow: hidden;
        }
        .hero-banner::after {
            content: "";
            position: absolute;
            top: -40%;
            right: -8%;
            width: 260px;
            height: 260px;
            background: radial-gradient(circle, rgba(59,130,246,0.25) 0%, rgba(59,130,246,0) 70%);
            border-radius: 50%;
        }
        .hero-icon {
            font-size: 2.2rem;
            margin-bottom: 0.3rem;
        }
        .hero-title {
            font-size: 1.9rem;
            font-weight: 800;
            color: #ffffff !important;
            letter-spacing: -0.02em;
            margin: 0;
            position: relative;
        }
        .hero-subtitle {
            font-size: 1rem;
            color: #93c5fd !important;
            font-weight: 500;
            margin-top: 0.35rem;
            position: relative;
        }

        hr.section-divider {
            margin: 1.6rem 0;
            border: none;
            border-top: 1px solid var(--border);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def recommendation_badge(label: str) -> str:
    mapping = {
        "Strong Match": "badge-strong",
        "Good Match": "badge-good",
        "Moderate Match": "badge-moderate",
        "Low Match": "badge-low",
    }
    css_class = mapping.get(label, "badge-good")
    return f'<span class="badge {css_class}">{label}</span>'


def score_color(score: int) -> str:
    if score >= 80:
        return "#4ade80"
    if score >= 60:
        return "#60a5fa"
    if score >= 40:
        return "#fbbf24"
    return "#f87171"


def score_bar(label: str, score: int) -> None:
    score = max(0, min(100, int(score)))
    color = score_color(score)
    st.markdown(
        f"""
        <div style="margin-bottom:0.6rem;">
            <div style="display:flex; justify-content:space-between; font-size:0.85rem; font-weight:600; color:var(--text-dark);">
                <span>{label}</span><span>{score}%</span>
            </div>
            <div style="background:#263450; border-radius:6px; height:9px; width:100%; margin-top:3px;">
                <div style="background:{color}; width:{score}%; height:9px; border-radius:6px;"></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ----------------------------------------------------------------------------
# DATABASE
# ----------------------------------------------------------------------------

@contextmanager
def get_conn():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    with get_conn() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS employees (
                employee_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                department TEXT,
                job_role TEXT,
                skills TEXT,
                joining_date TEXT,
                experience TEXT,
                status TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS candidates (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT,
                target_role TEXT,
                match_score INTEGER,
                recommendation TEXT,
                shortlisted INTEGER DEFAULT 0,
                created_at TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS performance_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                employee_name TEXT,
                tasks_completed INTEGER,
                attendance_pct REAL,
                manager_rating REAL,
                projects_completed INTEGER,
                summary TEXT,
                created_at TEXT
            )
            """
        )


def add_employee(emp: dict) -> tuple[bool, str]:
    try:
        with get_conn() as conn:
            conn.execute(
                """INSERT INTO employees
                (employee_id, name, department, job_role, skills, joining_date, experience, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    emp["employee_id"], emp["name"], emp["department"], emp["job_role"],
                    emp["skills"], emp["joining_date"], emp["experience"], emp["status"],
                ),
            )
        return True, "Employee added successfully."
    except sqlite3.IntegrityError:
        return False, "An employee with this Employee ID already exists."
    except Exception as e:
        return False, f"Could not add employee: {e}"


def get_employees(search: str = "") -> pd.DataFrame:
    with get_conn() as conn:
        if search:
            like = f"%{search}%"
            rows = conn.execute(
                """SELECT * FROM employees WHERE
                employee_id LIKE ? OR name LIKE ? OR department LIKE ? OR job_role LIKE ?""",
                (like, like, like, like),
            ).fetchall()
        else:
            rows = conn.execute("SELECT * FROM employees ORDER BY joining_date DESC").fetchall()
    return pd.DataFrame([dict(r) for r in rows])


def delete_employee(employee_id: str) -> None:
    with get_conn() as conn:
        conn.execute("DELETE FROM employees WHERE employee_id = ?", (employee_id,))


def log_candidate(name: str, target_role: str, match_score: int, recommendation: str) -> None:
    with get_conn() as conn:
        conn.execute(
            """INSERT INTO candidates (name, target_role, match_score, recommendation, created_at)
            VALUES (?, ?, ?, ?, ?)""",
            (name, target_role, match_score, recommendation, datetime.now().isoformat()),
        )


def get_candidates() -> pd.DataFrame:
    with get_conn() as conn:
        rows = conn.execute("SELECT * FROM candidates ORDER BY created_at DESC").fetchall()
    return pd.DataFrame([dict(r) for r in rows])


def save_performance_record(rec: dict) -> None:
    with get_conn() as conn:
        conn.execute(
            """INSERT INTO performance_records
            (employee_name, tasks_completed, attendance_pct, manager_rating, projects_completed, summary, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                rec["employee_name"], rec["tasks_completed"], rec["attendance_pct"],
                rec["manager_rating"], rec["projects_completed"], rec["summary"],
                datetime.now().isoformat(),
            ),
        )


# ----------------------------------------------------------------------------
# PDF EXTRACTION
# ----------------------------------------------------------------------------

def extract_pdf_text(uploaded_file) -> tuple[str, str]:
    """Returns (text, error_message). error_message is '' on success."""
    try:
        uploaded_file.seek(0)
        data = uploaded_file.read()
        if not data:
            return "", "The uploaded file appears to be empty."
        reader = PdfReader(io.BytesIO(data))
        if len(reader.pages) == 0:
            return "", "The PDF has no readable pages."

        text_parts = []
        for page in reader.pages:
            try:
                page_text = page.extract_text() or ""
                text_parts.append(page_text)
            except Exception:
                continue

        full_text = "\n".join(text_parts).strip()

        if not full_text or len(full_text) < 20:
            return "", (
                "No readable text was found in this PDF. It may be a scanned or "
                "image-only document. Please upload a text-based PDF resume."
            )
        return full_text, ""
    except Exception as e:
        return "", f"Could not read this PDF file. It may be corrupted. ({e})"


# ----------------------------------------------------------------------------
# GEMINI AI LAYER
# ----------------------------------------------------------------------------

def get_api_key() -> str:
    key = ""
    try:
        key = st.secrets.get("GEMINI_API_KEY", "")
    except Exception:
        key = ""
    if not key:
        key = os.environ.get("GEMINI_API_KEY", "")
    if not key:
        key = st.session_state.get("manual_api_key", "")
    return str(key).strip()


def get_model_name() -> str:
    try:
        secret_model = st.secrets.get("GEMINI_MODEL", "")
    except Exception:
        secret_model = ""
    return (
        st.session_state.get("manual_model_name")
        or secret_model
        or os.environ.get("GEMINI_MODEL", "")
        or DEFAULT_GEMINI_MODEL
    )


@st.cache_resource(show_spinner=False)
def _build_client(api_key: str):
    from google import genai
    return genai.Client(api_key=api_key)


def call_gemini(prompt: str, expect_json: bool = True, max_retries: int = 3) -> tuple[dict | str | None, str]:
    """
    Calls Gemini with the given prompt.
    Returns (parsed_result, error_message). If expect_json=False, parsed_result is raw text.
    error_message is '' on success.
    Automatically retries with backoff on transient errors (503 UNAVAILABLE / overloaded / 429 rate limit).
    """
    api_key = get_api_key()
    if not api_key:
        return None, "No Gemini API key configured. Please add one in the sidebar."

    try:
        client = _build_client(api_key)
    except Exception as e:
        return None, f"Could not initialize the Gemini client: {e}"

    model_name = get_model_name()

    from google.genai import types
    config = types.GenerateContentConfig(response_mime_type="application/json") if expect_json else None

    raw_text = None
    last_error_msg = ""

    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=config,
            )
            raw_text = getattr(response, "text", None)
            last_error_msg = ""
            break
        except Exception as e:
            err_str = str(e).lower()
            is_transient = (
                "503" in err_str or "unavailable" in err_str or "overloaded" in err_str
                or "429" in err_str or "rate" in err_str or "quota" in err_str
                or "timeout" in err_str or "deadline" in err_str
            )
            is_auth_error = (
                "api key" in err_str or "unauthorized" in err_str or "permission" in err_str
                or ("invalid" in err_str and "key" in err_str)
            )
            if is_auth_error:
                return None, "Your Gemini API key appears to be invalid or unauthorized. Please check it and try again."

            if is_transient and attempt < max_retries - 1:
                wait_seconds = 2 ** attempt  # 1s, 2s, 4s
                time.sleep(wait_seconds)
                last_error_msg = (
                    "The Gemini AI service is temporarily overloaded (high demand). "
                    "Please try again in a moment, or switch the model name in the sidebar's "
                    "Advanced settings if this keeps happening."
                )
                continue
            elif "quota" in err_str or "429" in err_str:
                return None, "The Gemini API rate limit or quota has been reached. Please wait a moment and try again."
            elif is_transient:
                return None, (
                    "The Gemini AI service is currently experiencing high demand and is temporarily "
                    "unavailable, even after a few retries. Please wait a minute and try again - this is "
                    "an issue on Google's side, not with the app."
                )
            else:
                return None, f"The AI service returned an error: {e}"

    if raw_text is None:
        return None, last_error_msg or "The AI service is temporarily unavailable. Please try again."

    if not raw_text or not raw_text.strip():
        return None, "The AI returned an empty response. Please try again."

    if not expect_json:
        return raw_text.strip(), ""

    parsed = safe_json_parse(raw_text)
    if parsed is None:
        # Fallback: surface the raw response instead of crashing.
        return {"_raw_fallback": raw_text.strip()}, ""
    return parsed, ""


def safe_json_parse(text: str) -> dict | list | None:
    """Robustly extract and parse JSON from an LLM response."""
    if not text:
        return None
    cleaned = text.strip()
    cleaned = re.sub(r"^```json\s*", "", cleaned)
    cleaned = re.sub(r"^```\s*", "", cleaned)
    cleaned = re.sub(r"```\s*$", "", cleaned)
    cleaned = cleaned.strip()

    try:
        return json.loads(cleaned)
    except Exception:
        pass

    # Try to locate the first {...} or [...] block
    for open_ch, close_ch in [("{", "}"), ("[", "]")]:
        start = cleaned.find(open_ch)
        end = cleaned.rfind(close_ch)
        if start != -1 and end != -1 and end > start:
            candidate = cleaned[start:end + 1]
            try:
                return json.loads(candidate)
            except Exception:
                continue
    return None


# ----------------------------------------------------------------------------
# PROMPT BUILDERS
# ----------------------------------------------------------------------------

def prompt_resume_analysis(resume_text: str, target_role: str) -> str:
    return f"""
You are an expert HR recruitment analyst. Analyze the resume text below and evaluate the
candidate strictly against the target job role: "{target_role}".

Resume text:
\"\"\"{resume_text[:12000]}\"\"\"

Return ONLY valid JSON (no markdown, no commentary) with EXACTLY this structure:
{{
  "candidate_name": "string, 'Not specified' if unclear",
  "education": ["string", "..."],
  "skills": ["string", "..."],
  "technical_skills": ["string", "..."],
  "soft_skills": ["string", "..."],
  "work_experience": ["string summary of each role", "..."],
  "projects": ["string", "..."],
  "certifications": ["string", "..."],
  "years_of_experience": "string, e.g. '3 years' or 'Fresher'",
  "key_strengths": ["string", "..."],
  "target_role": "{target_role}",
  "match_score": 0,
  "recommendation": "one of: Strong Match, Good Match, Moderate Match, Low Match",
  "why_this_role": ["short bullet string", "..."],
  "skill_gaps": ["short bullet string", "..."]
}}

match_score must be an integer 0-100 reflecting fit for "{target_role}" specifically.
If some information is missing from the resume, use an empty list or "Not specified" rather than inventing details.
"""


def prompt_best_fit_roles(resume_text: str, originally_selected_role: str) -> str:
    resume = resume_text[:12000]
    return f"""
You are an AI-powered HR Career Fit Analyst.

Analyze the candidate resume and provide a clear and professional career-fit recommendation for an HR professional.

This is decision SUPPORT only, not an automatic hiring decision.

The HR professional originally considered this candidate for: {originally_selected_role}

CANDIDATE RESUME:
\"\"\"{resume}\"\"\"

Analyze the candidate for multiple relevant roles, including:
Data Analyst, Business Analyst, HR Analytics, Financial Analyst, Marketing Analyst,
Software Developer, Operations, Sales, Customer Support, Product Management, Project Management.

Evaluate each role using education, technical skills, soft skills, projects, certifications,
internship or work experience, tools and technologies, and overall suitability.

SCORING SYSTEM:
90-100 = Excellent Fit
80-89 = Very Good Fit
70-79 = Good Fit
60-69 = Moderate Fit
Below 60 = Low Fit

IMPORTANT RULES:
- Base the analysis ONLY on information present in the resume.
- Do not invent skills, experience, qualifications, projects or certifications.
- Use simple professional HR language.
- Explanations must be easy for an HR professional to understand.
- Do not make a final hiring decision.
- The recommended role must be the strongest match based on the resume.

Return ONLY valid JSON. Do not return markdown. Do not add any text before or after the JSON.

Return EXACTLY this structure:
{{
  "candidate_name": "Candidate name or 'Candidate'",
  "candidate_summary": "A short 2-3 sentence professional summary of the candidate.",
  "recommended_role": "Single best-fit role",
  "recommended_role_score": 0,
  "recommended_role_explanation": "A clear 3-4 sentence explanation of why this role is the strongest fit.",
  "key_strengths": ["Important candidate strength", "Second important strength", "Third important strength"],
  "development_areas": ["Skill or area that can be improved", "Second development area", "Third development area"],
  "why_not_original_role": "A balanced explanation of how well the candidate fits the originally selected role and whether another role appears to be a stronger fit.",
  "roles": [
    {{
      "role": "Role name",
      "department": "Department name",
      "match_score": 0,
      "fit_level": "Excellent Fit",
      "matching_skills": ["Skill 1", "Skill 2", "Skill 3"],
      "missing_skills": ["Skill that would improve suitability", "Another useful skill"],
      "explanation": "A short 2-3 sentence explanation of the candidate's suitability for this role."
    }}
  ]
}}

REQUIREMENTS:
- Include at least 5 different roles.
- Include the originally selected role: {originally_selected_role}.
- Sort roles from highest match score to lowest.
- match_score must be an integer from 0 to 100.
- recommended_role_score must equal the match_score of the recommended role.
- matching_skills must contain only skills supported by the resume.
- missing_skills should contain realistic skills that would improve suitability.
- fit_level must be one of: Excellent Fit, Very Good Fit, Good Fit, Moderate Fit, Low Fit.
- Keep the language concise, professional and human-readable.
"""


def prompt_jd_match(resume_text: str, job_title: str, job_description: str) -> str:
    return f"""
You are an HR recruitment analyst. Compare the candidate resume against the job description below.

Job Title: {job_title}
Job Description:
\"\"\"{job_description[:6000]}\"\"\"

Resume text:
\"\"\"{resume_text[:12000]}\"\"\"

Return ONLY valid JSON (no markdown, no commentary) with EXACTLY this structure:
{{
  "overall_match_percentage": 0,
  "matching_skills": ["string"],
  "missing_skills": ["string"],
  "relevant_experience": ["string"],
  "strengths": ["string"],
  "weaknesses": ["string"],
  "recommendation": "one of: Strong Match, Good Match, Moderate Match, Low Match"
}}

overall_match_percentage must be an integer 0-100.
"""


def prompt_interview_questions(resume_text: str, target_role: str, job_description: str) -> str:
    jd_block = f'\nJob Description:\n"""{job_description[:4000]}"""\n' if job_description else ""
    return f"""
You are an experienced technical interviewer. Based ONLY on the candidate's resume, their skills,
the target role "{target_role}"{', and the job description below' if job_description else ''}, generate
interview questions. Do not generate questions unrelated to the candidate's resume.
{jd_block}
Resume text:
\"\"\"{resume_text[:12000]}\"\"\"

Return ONLY valid JSON (no markdown, no commentary) with EXACTLY this structure:
{{
  "technical_questions": ["string", "... exactly 5 items"],
  "behavioral_questions": ["string", "... exactly 3 items"],
  "resume_based_questions": ["string", "... exactly 2 items"]
}}
"""


def prompt_performance_summary(name: str, tasks: int, attendance: float, rating: float, projects: int) -> str:
    return f"""
You are an HR performance analyst. Based on the following employee performance inputs, write a
short, constructive performance summary.

Employee: {name}
Tasks completed: {tasks}
Attendance percentage: {attendance}%
Manager rating (out of 5): {rating}
Projects completed: {projects}

Return ONLY valid JSON (no markdown, no commentary) with EXACTLY this structure:
{{
  "overall_performance": "one short sentence/phrase, e.g. 'Excellent', 'Good', 'Needs Improvement'",
  "summary": "2-3 sentence overall summary",
  "strengths": ["string", "..."],
  "improvement_areas": ["string", "..."],
  "suggested_training": ["string", "..."]
}}
"""


# ----------------------------------------------------------------------------
# SESSION STATE
# ----------------------------------------------------------------------------

def init_session_state() -> None:
    defaults = {
        "resume_text": "",
        "resume_filename": "",
        "resume_analysis": None,
        "best_fit_analysis": None,
        "jd_match_result": None,
        "interview_questions": None,
        "selected_target_role": JOB_ROLES[0],
        "manual_api_key": "",
        "manual_model_name": "",
        "nav": "Dashboard",
        "pending_nav": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ----------------------------------------------------------------------------
# SIDEBAR
# ----------------------------------------------------------------------------

def render_sidebar() -> str:
    with st.sidebar:
        st.markdown('<p class="brand-title">💼 AI HR Assistant</p>', unsafe_allow_html=True)
        st.markdown(f'<p class="brand-sub">{APP_TAGLINE}</p>', unsafe_allow_html=True)

        nav = st.radio(
            "Navigation",
            [
                "🏠 Dashboard",
                "📄 Resume Analyzer",
                "🎯 Best-Fit Role",
                "🎤 Interview Questions",
                "👥 Employees",
                "📊 Performance Analysis",
            ],
            label_visibility="collapsed",
            key="nav_radio",
        )

        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("**AI Configuration**")

        st.text_input(
            "Gemini API Key",
            type="password",
            key="manual_api_key",
            placeholder="Enter your Gemini API key",
            help="Your API key is stored only for this session.",
        )

        has_key = bool(get_api_key())
        if has_key:
            st.success("Gemini API key detected", icon="✅")
        else:
            st.warning("Please enter your Gemini API key.")

        with st.expander("Advanced: Model settings"):
            st.text_input(
                "Gemini model name",
                key="manual_model_name",
                placeholder=DEFAULT_GEMINI_MODEL,
                help="Override the Gemini model used for AI calls (e.g. if the default model is overloaded).",
            )

        st.markdown("<hr>", unsafe_allow_html=True)
        st.caption("AI outputs are decision-support suggestions only. Final HR decisions remain with the human recruiter.")

    return nav


# ----------------------------------------------------------------------------
# PAGE: DASHBOARD
# ----------------------------------------------------------------------------

def page_dashboard() -> None:
    st.markdown(
        f"""
        <div class="hero-banner">
            <div class="hero-icon">💼</div>
            <p class="hero-title">{APP_TITLE}</p>
            <p class="hero-subtitle">{APP_TAGLINE}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown('<p class="page-header">Dashboard</p>', unsafe_allow_html=True)
    st.markdown('<p class="page-subtext">Overview of recruitment and workforce activity.</p>', unsafe_allow_html=True)

    candidates_df = get_candidates()
    employees_df = get_employees()

    total_candidates = len(candidates_df)
    shortlisted = int((candidates_df["match_score"] >= 70).sum()) if not candidates_df.empty else 0
    total_employees = len(employees_df)
    open_positions = len(JOB_ROLES)

    c1, c2, c3, c4 = st.columns(4)
    metrics = [
        (c1, "Total Candidates", total_candidates),
        (c2, "Shortlisted Candidates", shortlisted),
        (c3, "Total Employees", total_employees),
        (c4, "Open Positions", open_positions),
    ]
    for col, label, value in metrics:
        with col:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">{label}</div>
                    <div class="metric-value">{value}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

    # ---- Quick Actions: clickable icon tiles that jump straight to a section ----
    st.markdown("##### Quick Actions")
    tile_defs = [
        ("📄", "Resume Analyzer", "📄 Resume Analyzer"),
        ("🎯", "Best-Fit Role", "🎯 Best-Fit Role"),
        ("🎤", "Interview Questions", "🎤 Interview Questions"),
        ("👥", "Employees", "👥 Employees"),
        ("📊", "Performance Analysis", "📊 Performance Analysis"),
    ]
    tile_cols = st.columns(len(tile_defs))
    for col, (icon, label, nav_value) in zip(tile_cols, tile_defs):
        with col:
            st.markdown('<div class="tile-btn">', unsafe_allow_html=True)
            if st.button(f"{icon}  {label}", key=f"tile_{label}", use_container_width=True):
                st.session_state["pending_nav"] = nav_value
                st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

    col_left, col_right = st.columns(2)

    chart_layout = dict(
        plot_bgcolor="#131c2e", paper_bgcolor="#131c2e",
        font=dict(color="#f1f5f9"),
        margin=dict(l=10, r=10, t=10, b=10), height=350,
    )

    with col_left:
        st.markdown("##### Candidate Match Scores")
        if candidates_df.empty:
            st.info("No candidates analyzed yet. Head to Resume Analyzer to get started.")
        else:
            recent = candidates_df.head(10).iloc[::-1]
            fig = px.bar(
                recent, x="match_score", y="name", orientation="h",
                color="match_score", color_continuous_scale=["#f87171", "#fbbf24", "#3b82f6", "#4ade80"],
                range_color=[0, 100], labels={"match_score": "Match %", "name": "Candidate"},
            )
            fig.update_layout(showlegend=False, coloraxis_showscale=False, **chart_layout)
            fig.update_xaxes(gridcolor="#263450", color="#f1f5f9")
            fig.update_yaxes(gridcolor="#263450", color="#f1f5f9")
            st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.markdown("##### Employees by Department")
        if employees_df.empty:
            st.info("No employees added yet. Head to the Employees section.")
        else:
            dept_counts = employees_df["department"].value_counts().reset_index()
            dept_counts.columns = ["department", "count"]
            fig2 = px.pie(
                dept_counts, names="department", values="count", hole=0.55,
                color_discrete_sequence=["#3b82f6", "#60a5fa", "#94a3b8", "#eab308", "#4ade80", "#f87171"],
            )
            fig2.update_layout(**chart_layout)
            fig2.update_traces(textfont=dict(color="#0b1120"))
            st.plotly_chart(fig2, use_container_width=True)


# ----------------------------------------------------------------------------
# PAGE: RESUME ANALYZER
# ----------------------------------------------------------------------------

def page_resume_analyzer() -> None:
    st.markdown('<p class="page-header">AI Resume Analyzer</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="page-subtext">Upload a candidate resume and let AI analyze their skills, experience and best-fit roles.</p>',
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns([2, 1])
    with col1:
        uploaded_file = st.file_uploader("Upload Resume (PDF)", type=["pdf"], key="resume_upload")
    with col2:
        target_role = st.selectbox("Target Job Role", JOB_ROLES, key="selected_target_role")

    if uploaded_file is not None:
        if uploaded_file.name != st.session_state.get("resume_filename"):
            text, err = extract_pdf_text(uploaded_file)
            if err:
                st.error(err)
                st.session_state.resume_text = ""
                st.session_state.resume_analysis = None
            else:
                st.session_state.resume_text = text
                st.session_state.resume_filename = uploaded_file.name
                st.session_state.resume_analysis = None
                st.session_state.best_fit_analysis = None
                st.session_state.interview_questions = None

    analyze_disabled = not st.session_state.resume_text
    if st.button("🔍 Analyze Resume", disabled=analyze_disabled, key="btn_analyze_resume"):
        with st.spinner("AI is analyzing the resume..."):
            result, err = call_gemini(
                prompt_resume_analysis(st.session_state.resume_text, target_role), expect_json=True
            )
        if err:
            st.error(err)
        elif result and "_raw_fallback" in result:
            st.warning("The AI response could not be parsed as structured data. Showing raw output below.")
            st.text(result["_raw_fallback"])
        elif result:
            st.session_state.resume_analysis = result
            try:
                log_candidate(
                    result.get("candidate_name", "Unknown"),
                    result.get("target_role", target_role),
                    int(result.get("match_score", 0)),
                    result.get("recommendation", "Moderate Match"),
                )
            except Exception:
                pass

    analysis = st.session_state.resume_analysis
    if analysis and "_raw_fallback" not in analysis:
        st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

        top1, top2, top3 = st.columns(3)
        with top1:
            st.markdown(
                f"""<div class="hr-card"><div class="metric-label">Candidate</div>
                <div style="font-size:1.3rem;font-weight:700;color:var(--accent-light);">{analysis.get('candidate_name', 'Not specified')}</div></div>""",
                unsafe_allow_html=True,
            )
        with top2:
            st.markdown(
                f"""<div class="hr-card"><div class="metric-label">Target Role</div>
                <div style="font-size:1.3rem;font-weight:700;color:var(--accent-light);">{analysis.get('target_role', target_role)}</div></div>""",
                unsafe_allow_html=True,
            )
        with top3:
            rec = analysis.get("recommendation", "Moderate Match")
            st.markdown(
                f"""<div class="hr-card"><div class="metric-label">Recommendation</div>
                <div style="margin-top:0.3rem;">{recommendation_badge(rec)}</div></div>""",
                unsafe_allow_html=True,
            )

        st.markdown("<div class='hr-card'>", unsafe_allow_html=True)
        score_bar("Match Score", analysis.get("match_score", 0))
        st.markdown("</div>", unsafe_allow_html=True)

        colA, colB = st.columns(2)
        with colA:
            with st.container():
                st.markdown("<div class='hr-card'>", unsafe_allow_html=True)
                st.markdown("**Why this role?**")
                for item in analysis.get("why_this_role", []):
                    st.markdown(f"- {item}")
                st.markdown("</div>", unsafe_allow_html=True)

            with st.container():
                st.markdown("<div class='hr-card'>", unsafe_allow_html=True)
                st.markdown("**Education**")
                for item in analysis.get("education", []) or ["Not specified"]:
                    st.markdown(f"- {item}")
                st.markdown("</div>", unsafe_allow_html=True)

            with st.container():
                st.markdown("<div class='hr-card'>", unsafe_allow_html=True)
                st.markdown("**Work Experience**")
                for item in analysis.get("work_experience", []) or ["Not specified"]:
                    st.markdown(f"- {item}")
                st.markdown("</div>", unsafe_allow_html=True)

        with colB:
            with st.container():
                st.markdown("<div class='hr-card'>", unsafe_allow_html=True)
                st.markdown("**Skill Gaps**")
                gaps = analysis.get("skill_gaps", [])
                if gaps:
                    for item in gaps:
                        st.markdown(f"- {item}")
                else:
                    st.markdown("- No major gaps identified")
                st.markdown("</div>", unsafe_allow_html=True)

            with st.container():
                st.markdown("<div class='hr-card'>", unsafe_allow_html=True)
                st.markdown("**Technical / Soft Skills**")
                st.markdown("*Technical:* " + (", ".join(analysis.get("technical_skills", [])) or "Not specified"))
                st.markdown("*Soft skills:* " + (", ".join(analysis.get("soft_skills", [])) or "Not specified"))
                st.markdown("</div>", unsafe_allow_html=True)

            with st.container():
                st.markdown("<div class='hr-card'>", unsafe_allow_html=True)
                st.markdown("**Projects & Certifications**")
                st.markdown("*Projects:* " + (", ".join(analysis.get("projects", [])) or "Not specified"))
                st.markdown("*Certifications:* " + (", ".join(analysis.get("certifications", [])) or "Not specified"))
                st.markdown(f"*Experience:* {analysis.get('years_of_experience', 'Not specified')}")
                st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='hr-card'>", unsafe_allow_html=True)
        st.markdown("**Key Strengths**")
        strengths = analysis.get("key_strengths", [])
        if strengths:
            st.markdown(" &nbsp;•&nbsp; ".join(strengths))
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
        st.markdown("##### Job Description Matching (optional)")
        job_title_input = st.text_input("Job Title", value=analysis.get("target_role", target_role), key="jd_title")
        jd_text = st.text_area(
            "Paste the full Job Description (required skills, responsibilities, etc.)",
            height=150, key="jd_text_input",
        )
        if st.button("📋 Match Against Job Description", key="btn_jd_match"):
            if not jd_text.strip():
                st.warning("Please paste a job description first.")
            else:
                with st.spinner("Comparing resume against job description..."):
                    jd_result, jd_err = call_gemini(
                        prompt_jd_match(st.session_state.resume_text, job_title_input, jd_text), expect_json=True
                    )
                if jd_err:
                    st.error(jd_err)
                elif jd_result and "_raw_fallback" in jd_result:
                    st.warning("Could not parse structured output. Raw AI response:")
                    st.text(jd_result["_raw_fallback"])
                else:
                    st.session_state.jd_match_result = jd_result

        jd_result = st.session_state.jd_match_result
        if jd_result and "_raw_fallback" not in jd_result:
            st.markdown("<div class='hr-card'>", unsafe_allow_html=True)
            score_bar("Overall Match", jd_result.get("overall_match_percentage", 0))
            st.markdown(recommendation_badge(jd_result.get("recommendation", "Moderate Match")), unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

            jc1, jc2 = st.columns(2)
            with jc1:
                st.markdown("<div class='hr-card'>", unsafe_allow_html=True)
                st.markdown("**✅ Matching Skills**")
                st.markdown(", ".join(jd_result.get("matching_skills", [])) or "None identified")
                st.markdown("**Relevant Experience**")
                for item in jd_result.get("relevant_experience", []):
                    st.markdown(f"- {item}")
                st.markdown("</div>", unsafe_allow_html=True)
            with jc2:
                st.markdown("<div class='hr-card'>", unsafe_allow_html=True)
                st.markdown("**❌ Missing Skills**")
                st.markdown(", ".join(jd_result.get("missing_skills", [])) or "None identified")
                st.markdown("**Weaknesses / Gaps**")
                for item in jd_result.get("weaknesses", []):
                    st.markdown(f"- {item}")
                st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
        st.markdown("##### Add to Employee Database")
        with st.form("add_candidate_form"):
            fc1, fc2, fc3 = st.columns(3)
            with fc1:
                emp_id = st.text_input("Employee ID", value=f"EMP{datetime.now().strftime('%y%m%d%H%M')}")
                emp_name = st.text_input("Name", value=analysis.get("candidate_name", ""))
            with fc2:
                emp_dept = st.text_input("Department", value=analysis.get("target_role", target_role))
                emp_role = st.text_input("Job Role", value=target_role)
            with fc3:
                emp_join = st.date_input("Joining Date", value=date.today())
                emp_status = st.selectbox("Status", STATUS_OPTIONS, index=0)
            emp_skills = st.text_input(
                "Skills", value=", ".join(analysis.get("technical_skills", []) + analysis.get("soft_skills", []))
            )
            emp_exp = st.text_input("Experience", value=analysis.get("years_of_experience", ""))
            submitted = st.form_submit_button("➕ Add as Employee")
            if submitted:
                if not emp_id.strip() or not emp_name.strip():
                    st.warning("Employee ID and Name are required.")
                else:
                    ok, msg = add_employee({
                        "employee_id": emp_id.strip(), "name": emp_name.strip(),
                        "department": emp_dept, "job_role": emp_role, "skills": emp_skills,
                        "joining_date": str(emp_join), "experience": emp_exp, "status": emp_status,
                    })
                    if ok:
                        st.success(msg)
                    else:
                        st.error(msg)


# ----------------------------------------------------------------------------
# PAGE: BEST-FIT ROLE
# ----------------------------------------------------------------------------

def _coerce_score(role: dict) -> float:
    """Safely extract a numeric match score from a role dict, tolerating strings like '85%'."""
    value = role.get("match_score", 0)
    try:
        if isinstance(value, str):
            value = value.replace("%", "").strip()
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def page_best_fit_role() -> None:
    st.markdown('<p class="page-header">AI Career Fit Analysis</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="page-subtext">AI-powered recommendation for HR decision support — '
        'the final hiring decision remains with you.</p>',
        unsafe_allow_html=True,
    )

    if not st.session_state.get("resume_text", "").strip():
        st.info("Please upload and analyze a resume first in the Resume Analyzer section.")
        return

    if st.button("🎯 Find Best-Fit Role / Department", key="btn_best_fit"):
        with st.spinner("AI is evaluating the candidate across roles..."):
            result, err = call_gemini(
                prompt_best_fit_roles(
                    st.session_state.resume_text,
                    st.session_state.get("selected_target_role", JOB_ROLES[0]),
                ),
                expect_json=True,
            )
        if err:
            st.error(err)
        elif result is None:
            st.error("The AI did not return a response. Please try again.")
        else:
            if isinstance(result, list):
                result = {"roles": result}
            st.session_state.best_fit_analysis = result

    result = st.session_state.get("best_fit_analysis")
    if result is None:
        return

    if isinstance(result, list):
        result = {"roles": result}

    if not isinstance(result, dict):
        st.error("The AI returned an unexpected response format.")
        return

    if "_raw_fallback" in result:
        st.warning("The AI response could not be structured automatically.")
        with st.expander("View AI response"):
            st.text(str(result.get("_raw_fallback", "")))
        return

    # ---- Extract & normalize roles ----
    raw_roles = result.get("roles", [])
    if isinstance(raw_roles, dict):
        raw_roles = [raw_roles]
    if not isinstance(raw_roles, list):
        st.error("The AI returned an invalid role list.")
        return

    roles = [item for item in raw_roles if isinstance(item, dict)]
    if not roles:
        st.warning("No suitable roles were returned by the AI. Please try the analysis again.")
        return

    roles = sorted(roles, key=_coerce_score, reverse=True)

    # ---- Candidate profile ----
    candidate_name = result.get("candidate_name", "Candidate")
    candidate_summary = result.get("candidate_summary", "")
    if not isinstance(candidate_summary, str):
        candidate_summary = str(candidate_summary)

    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
    st.markdown("### 👤 Candidate Profile")

    if candidate_summary:
        st.markdown(
            f"""
            <div class="hr-card">
                <div style="font-size:1.25rem; font-weight:800; margin-bottom:8px;">{candidate_name}</div>
                <div style="color:var(--text-muted); line-height:1.6;">{candidate_summary}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(f"**Candidate:** {candidate_name}")

    # ---- Recommended role ----
    recommended_role = result.get("recommended_role", "") or roles[0].get("role", "Recommended Role")

    recommended_score = result.get("recommended_role_score", _coerce_score(roles[0]))
    try:
        if isinstance(recommended_score, str):
            recommended_score = recommended_score.replace("%", "").strip()
        recommended_score = float(recommended_score)
    except (TypeError, ValueError):
        recommended_score = _coerce_score(roles[0])

    recommended_explanation = result.get("recommended_role_explanation", "")
    if not isinstance(recommended_explanation, str):
        recommended_explanation = str(recommended_explanation)

    st.markdown("### 🏆 Recommended Role")
    st.markdown(
        f"""
        <div class="hr-card" style="border-left:5px solid #4ade80; padding:22px;">
            <div style="font-size:1.55rem; font-weight:800; margin-bottom:6px;">{recommended_role}</div>
            <div style="font-size:1.15rem; font-weight:700; margin-bottom:12px;">{recommended_score:g}% Match</div>
            <div style="color:var(--text-muted); line-height:1.7;">{recommended_explanation}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ---- Key strengths ----
    strengths = result.get("key_strengths", [])
    if isinstance(strengths, str):
        strengths = [strengths]
    if not isinstance(strengths, list):
        strengths = []

    if strengths:
        st.markdown("### 💪 Key Strengths")
        cols = st.columns(min(len(strengths), 3))
        for index, strength in enumerate(strengths[:3]):
            with cols[index]:
                st.markdown(
                    f"""
                    <div class="hr-card" style="min-height:100px;">
                        <div style="font-size:1.5rem; margin-bottom:6px;">✓</div>
                        <div style="line-height:1.5;">{strength}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    # ---- Development areas ----
    development = result.get("development_areas", [])
    if isinstance(development, str):
        development = [development]
    if not isinstance(development, list):
        development = []

    if development:
        st.markdown("### 📈 Development Areas")
        for item in development[:4]:
            st.markdown(f"- {item}")

    # ---- Why not original role ----
    selected_role = st.session_state.get("selected_target_role", "Selected Role")
    why_not_original = result.get("why_not_original_role", "")
    if why_not_original:
        if not isinstance(why_not_original, str):
            why_not_original = str(why_not_original)
        with st.expander(f"💡 Fit against the originally selected role: {selected_role}"):
            st.write(why_not_original)

    # ---- Role compatibility cards ----
    st.markdown("### 📊 Role Compatibility")
    for index, role in enumerate(roles):
        role_name = str(role.get("role", "Unknown Role"))
        department = str(role.get("department", ""))
        score = _coerce_score(role)

        fit_level = role.get("fit_level", "")
        if not fit_level:
            if score >= 90:
                fit_level = "Excellent Fit"
            elif score >= 80:
                fit_level = "Very Good Fit"
            elif score >= 70:
                fit_level = "Good Fit"
            elif score >= 60:
                fit_level = "Moderate Fit"
            else:
                fit_level = "Low Fit"

        matching = role.get("matching_skills", [])
        if isinstance(matching, str):
            matching = [matching]
        if not isinstance(matching, list):
            matching = []

        missing = role.get("missing_skills", [])
        if isinstance(missing, str):
            missing = [missing]
        if not isinstance(missing, list):
            missing = []

        explanation = role.get("explanation", "")
        if not isinstance(explanation, str):
            explanation = str(explanation)

        if index == 0:
            rank = "🥇"
        elif index == 1:
            rank = "🥈"
        elif index == 2:
            rank = "🥉"
        else:
            rank = f"{index + 1}."

        st.markdown(
            f"""
            <div class="hr-card" style="margin-bottom:15px; padding:20px;">
                <div style="display:flex; justify-content:space-between; align-items:center; gap:15px;">
                    <div>
                        <div style="font-size:1.15rem; font-weight:800;">{rank} {role_name}</div>
                        <div style="color:var(--text-muted); margin-top:4px;">{department}</div>
                    </div>
                    <div style="font-size:1.25rem; font-weight:800;">{score:g}%</div>
                </div>
                <div style="margin-top:12px; font-weight:700;">{fit_level}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if matching:
            st.markdown("**✅ Matching Skills:** " + ", ".join(str(x) for x in matching))
        if missing:
            st.markdown("**📚 Skills to Develop:** " + ", ".join(str(x) for x in missing))
        if explanation:
            st.markdown(f"**Why this fit?** {explanation}")
        st.markdown("<br>", unsafe_allow_html=True)

    # ---- Summary table ----
    st.markdown("### 📋 Quick Comparison")
    table_data = []
    for role in roles:
        score = _coerce_score(role)
        table_data.append({
            "Role": str(role.get("role", "Unknown")),
            "Department": str(role.get("department", "—")),
            "Match": f"{score:g}%",
            "Fit": str(role.get("fit_level", "—")),
        })

    if table_data:
        table_df = pd.DataFrame(table_data)
        st.dataframe(table_df, use_container_width=True, hide_index=True)


# ----------------------------------------------------------------------------
# PAGE: INTERVIEW QUESTIONS
# ----------------------------------------------------------------------------

def page_interview_questions() -> None:
    st.markdown('<p class="page-header">AI Interview Question Generator</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="page-subtext">Generate tailored interview questions based on the candidate\'s resume and target role.</p>',
        unsafe_allow_html=True,
    )

    if not st.session_state.resume_text:
        st.info("Please upload and analyze a resume first in the Resume Analyzer section.")
        return

    role = st.session_state.selected_target_role
    if st.session_state.resume_analysis and "_raw_fallback" not in (st.session_state.resume_analysis or {}):
        role = st.session_state.resume_analysis.get("target_role", role)

    st.markdown(f"**Target Role:** {role}")
    jd_context = ""
    if st.session_state.jd_match_result:
        jd_context = st.session_state.get("jd_text_input", "") or ""

    if st.button("🎤 Generate Interview Questions", key="btn_gen_questions"):
        with st.spinner("AI is generating interview questions..."):
            result, err = call_gemini(
                prompt_interview_questions(st.session_state.resume_text, role, jd_context), expect_json=True
            )
        if err:
            st.error(err)
        elif result and "_raw_fallback" in result:
            st.warning("Could not parse structured output. Raw AI response:")
            st.text(result["_raw_fallback"])
        else:
            st.session_state.interview_questions = result

    questions = st.session_state.interview_questions
    if questions and "_raw_fallback" not in questions:
        st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

        st.markdown("<div class='hr-card'>", unsafe_allow_html=True)
        st.markdown("**🛠️ Technical Questions**")
        for i, q in enumerate(questions.get("technical_questions", []), 1):
            st.markdown(f"{i}. {q}")
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='hr-card'>", unsafe_allow_html=True)
        st.markdown("**🤝 Behavioral / Situational Questions**")
        for i, q in enumerate(questions.get("behavioral_questions", []), 1):
            st.markdown(f"{i}. {q}")
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='hr-card'>", unsafe_allow_html=True)
        st.markdown("**📄 Resume-Based Questions**")
        for i, q in enumerate(questions.get("resume_based_questions", []), 1):
            st.markdown(f"{i}. {q}")
        st.markdown("</div>", unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# PAGE: EMPLOYEES
# ----------------------------------------------------------------------------

def page_employees() -> None:
    st.markdown('<p class="page-header">Employee Management</p>', unsafe_allow_html=True)
    st.markdown('<p class="page-subtext">Add, view, search and manage employee records.</p>', unsafe_allow_html=True)

    tab_view, tab_add = st.tabs(["👥 View & Search", "➕ Add Employee"])

    with tab_add:
        with st.form("manual_add_employee_form"):
            c1, c2, c3 = st.columns(3)
            with c1:
                emp_id = st.text_input("Employee ID*")
                emp_name = st.text_input("Name*")
            with c2:
                emp_dept = st.text_input("Department")
                emp_role = st.text_input("Job Role")
            with c3:
                emp_join = st.date_input("Joining Date", value=date.today())
                emp_status = st.selectbox("Status", STATUS_OPTIONS, index=0)
            emp_skills = st.text_input("Skills (comma separated)")
            emp_exp = st.text_input("Experience (e.g. '2 years')")
            submitted = st.form_submit_button("Add Employee")
            if submitted:
                if not emp_id.strip() or not emp_name.strip():
                    st.warning("Employee ID and Name are required fields.")
                else:
                    ok, msg = add_employee({
                        "employee_id": emp_id.strip(), "name": emp_name.strip(),
                        "department": emp_dept, "job_role": emp_role, "skills": emp_skills,
                        "joining_date": str(emp_join), "experience": emp_exp, "status": emp_status,
                    })
                    if ok:
                        st.success(msg)
                    else:
                        st.error(msg)

    with tab_view:
        search_term = st.text_input("🔍 Search by ID, name, department or role", key="emp_search")
        df = get_employees(search_term)
        if df.empty:
            st.info("No employees found." if search_term else "No employees added yet.")
        else:
            st.dataframe(df, use_container_width=True, hide_index=True)

            st.markdown("##### Remove an Employee")
            del_id = st.selectbox(
                "Select Employee ID to delete", options=["-- select --"] + df["employee_id"].tolist(),
                key="delete_emp_select",
            )
            if del_id != "-- select --":
                if st.button(f"🗑️ Delete {del_id}", key="btn_delete_emp"):
                    delete_employee(del_id)
                    st.success(f"Employee {del_id} deleted.")
                    st.rerun()


# ----------------------------------------------------------------------------
# PAGE: PERFORMANCE ANALYSIS
# ----------------------------------------------------------------------------

def page_performance_analysis() -> None:
    st.markdown('<p class="page-header">Employee Performance Analysis</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="page-subtext">Enter performance data and let AI generate a short summary and recommendations.</p>',
        unsafe_allow_html=True,
    )

    with st.form("performance_form"):
        c1, c2 = st.columns(2)
        with c1:
            name = st.text_input("Employee Name*")
            tasks = st.number_input("Tasks Completed", min_value=0, step=1, value=0)
        with c2:
            attendance = st.slider("Attendance Percentage", 0, 100, 90)
            rating = st.slider("Manager Rating (out of 5)", 0.0, 5.0, 3.5, step=0.1)
        projects = st.number_input("Projects Completed", min_value=0, step=1, value=0)
        submitted = st.form_submit_button("📊 Generate Performance Summary")

    if submitted:
        if not name.strip():
            st.warning("Please enter the employee's name.")
        else:
            with st.spinner("AI is generating the performance summary..."):
                result, err = call_gemini(
                    prompt_performance_summary(name, int(tasks), float(attendance), float(rating), int(projects)),
                    expect_json=True,
                )
            if err:
                st.error(err)
            elif result and "_raw_fallback" in result:
                st.warning("Could not parse structured output. Raw AI response:")
                st.text(result["_raw_fallback"])
            else:
                st.session_state["perf_result"] = result
                st.session_state["perf_name"] = name
                try:
                    save_performance_record({
                        "employee_name": name, "tasks_completed": int(tasks),
                        "attendance_pct": float(attendance), "manager_rating": float(rating),
                        "projects_completed": int(projects), "summary": result.get("summary", ""),
                    })
                except Exception:
                    pass

    result = st.session_state.get("perf_result")
    if result:
        st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
        st.markdown("<div class='hr-card'>", unsafe_allow_html=True)
        st.markdown(f"**Employee:** {st.session_state.get('perf_name','')}")
        st.markdown(f"**Overall Performance:** {result.get('overall_performance','')}")
        st.markdown(result.get("summary", ""))
        st.markdown("</div>", unsafe_allow_html=True)

        pc1, pc2, pc3 = st.columns(3)
        with pc1:
            st.markdown("<div class='hr-card'>", unsafe_allow_html=True)
            st.markdown("**Strengths**")
            for item in result.get("strengths", []):
                st.markdown(f"- {item}")
            st.markdown("</div>", unsafe_allow_html=True)
        with pc2:
            st.markdown("<div class='hr-card'>", unsafe_allow_html=True)
            st.markdown("**Improvement Areas**")
            for item in result.get("improvement_areas", []):
                st.markdown(f"- {item}")
            st.markdown("</div>", unsafe_allow_html=True)
        with pc3:
            st.markdown("<div class='hr-card'>", unsafe_allow_html=True)
            st.markdown("**Suggested Training**")
            for item in result.get("suggested_training", []):
                st.markdown(f"- {item}")
            st.markdown("</div>", unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------------

def main() -> None:
    inject_css()
    init_session_state()
    init_db()

    # Apply any navigation change requested by a Quick Action tile BEFORE the
    # sidebar radio widget is instantiated below - a widget's session_state
    # key cannot be reassigned after it has already been created in the
    # same script run, so we resolve pending navigation here first.
    pending_nav = st.session_state.pop("pending_nav", None)
    if pending_nav:
        st.session_state["nav_radio"] = pending_nav

    nav = render_sidebar()

    page_map = {
        "🏠 Dashboard": page_dashboard,
        "📄 Resume Analyzer": page_resume_analyzer,
        "🎯 Best-Fit Role": page_best_fit_role,
        "🎤 Interview Questions": page_interview_questions,
        "👥 Employees": page_employees,
        "📊 Performance Analysis": page_performance_analysis,
    }
    page_map.get(nav, page_dashboard)()


if __name__ == "__main__":
    main()
