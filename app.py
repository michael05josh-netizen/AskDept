import os
import json
import re
import streamlit as st
import time
from dotenv import load_dotenv

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.messages import AIMessageChunk

# =====================================================
# ENV SETUP
# =====================================================
load_dotenv()

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# =====================================================
# STREAMLIT CONFIG + MOBILE UI IMPROVEMENTS
# =====================================================
st.set_page_config(
    page_title="AskDept",
    page_icon="",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# Improved CSS - Fixed Top Cut-off Issue
st.markdown("""
<style>
    /* Fix top cut-off */
    .block-container {
        padding-top: 1.2rem !important;
        padding-bottom: 1rem;
    }

    /* Main title styling */
    h1 {
        font-size: 1.85rem !important;
        margin-top: 0.3rem !important;
        margin-bottom: 0.4rem !important;
        font-weight: 700;
        text-align: center;
    }

    /* Mobile adjustments */
    @media (max-width: 768px) {
        h1 {
            font-size: 1.65rem !important;
            margin-top: 0.5rem !important;
        }
        .block-container {
            padding-top: 1rem !important;
        }
    }

    .stChatInput {
        padding-bottom: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)

# =====================================================
# SPLASH SCREEN (2 seconds)
# =====================================================
if "splash_shown" not in st.session_state:
    st.markdown("""
    <style>
    .splash {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100vh;
        background: linear-gradient(135deg, #1e40af, #3b82f6);
        display: flex;
        align-items: center;
        justify-content: center;
        z-index: 9999;
        color: white;
    }
    .splash-content {
        text-align: center;
    }
    .splash h1 {
        font-size: 3.8rem;
        margin: 0 0 12px 0;
        font-weight: 700;
    }
    .splash p {
        font-size: 1.25rem;
        opacity: 0.95;
    }
    </style>
    
    <div class="splash">
        <div class="splash-content">
            <h1>AskDept</h1>
            <p>Computer Engineering Department</p>
            <p>AI Assistant</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    time.sleep(2)
    st.session_state.splash_shown = True
    st.rerun()

# =====================================================
# HEADER
# =====================================================
st.markdown("<h1>AskDept 🏫</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #555; margin-top: -8px;'>Computer Engineering • ABU Zaria</p>", unsafe_allow_html=True)

st.markdown("---")

# =====================================================
# SESSION STATE
# =====================================================
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "eval_log" not in st.session_state:
    st.session_state.eval_log = []
if "splash_shown" not in st.session_state:
    st.session_state.splash_shown = False

# =====================================================
# PATHS
# =====================================================
BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
DATA_DIR       = os.path.join(BASE_DIR, "data")
CURRICULUM_DIR = os.path.join(DATA_DIR, "curriculum")
STRUCTURED_DIR = os.path.join(DATA_DIR, "structured")
RAG_DOCS_DIR   = os.path.join(DATA_DIR, "rag_docs")
PERSIST_DIR    = os.path.join(BASE_DIR, "vectorstore")

# =====================================================
# DATA LOADERS
# =====================================================

@st.cache_resource
@st.cache_resource
def load_curriculum() -> dict:
    """
    Loads all curriculum JSON files, keyed by (level, semester, option).
    Files with no specific option (e.g. 400_second_common.json) are also
    registered under every option that exists at the same level, so a
    lookup like (400, second, AI) finds the shared courses.
    """
    curriculum = {}
    if not os.path.exists(CURRICULUM_DIR):
        return curriculum

    for filename in os.listdir(CURRICULUM_DIR):
        if not filename.lower().endswith(".json"):
            continue
        path = os.path.join(CURRICULUM_DIR, filename)
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            level    = str(data.get("level", "")).strip()
            semester = str(data.get("semester", "")).strip().lower()
            option   = data.get("option", None)
            if level and semester:
                curriculum[(level, semester, option)] = data
        except Exception as e:
            st.warning(f"Could not load {filename}: {e}")

    # --- Share "common" files across all options at the same level ---
    COMMON_MARKERS = {None, "", "common", "all", "general"}

    def is_common(opt):
        return opt is None or str(opt).strip().lower() in COMMON_MARKERS

    # Real options per level, taken from the data itself
    options_by_level = {}
    for (lvl, sem, opt) in list(curriculum.keys()):
        if not is_common(opt):
            options_by_level.setdefault(lvl, set()).add(opt)

    for (lvl, sem, opt), data in list(curriculum.items()):
        if is_common(opt):
            for real_opt in options_by_level.get(lvl, []):
                curriculum.setdefault((lvl, sem, real_opt), data)

    return curriculum


@st.cache_resource
def load_classrooms() -> list:
    path = os.path.join(STRUCTURED_DIR, "classrooms.json")
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("rooms", [])
    except Exception as e:
        st.warning(f"Could not load classrooms.json: {e}")
        return []


@st.cache_resource
def load_offices() -> list:
    path = os.path.join(STRUCTURED_DIR, "offices.json")
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("offices", [])
    except Exception as e:
        st.warning(f"Could not load offices.json: {e}")
        return []


# Load databases
curriculum_db = load_curriculum()
classrooms_db = load_classrooms()
offices_db    = load_offices()

# =====================================================
# VECTOR STORE
# =====================================================

@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

embeddings = get_embeddings()


@st.cache_resource
def load_or_create_vectorstore():
    if os.path.exists(PERSIST_DIR) and os.listdir(PERSIST_DIR):
        return FAISS.load_local(
            PERSIST_DIR, embeddings, allow_dangerous_deserialization=True
        )
    
    all_docs = []
    if os.path.exists(RAG_DOCS_DIR):
        for fname in os.listdir(RAG_DOCS_DIR):
            if fname.lower().endswith(".pdf"):
                try:
                    loader = PyPDFLoader(os.path.join(RAG_DOCS_DIR, fname))
                    docs = loader.load()
                    for d in docs:
                        d.metadata["source"] = fname
                        d.metadata["type"] = "pdf"
                    all_docs.extend(docs)
                except Exception as e:
                    st.warning(f"Could not load {fname}: {e}")
    
    if not all_docs:
        from langchain_core.documents import Document
        all_docs = [Document(page_content="placeholder", metadata={})]
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = splitter.split_documents(all_docs)
    db = FAISS.from_documents(splits, embeddings)
    
    os.makedirs(PERSIST_DIR, exist_ok=True)
    db.save_local(PERSIST_DIR)
    return db


vector_db = load_or_create_vectorstore()
retriever = vector_db.as_retriever(
    search_type="mmr", 
    search_kwargs={"k": 8, "fetch_k": 20}
)

# =====================================================
# LLM SETUP
# =====================================================
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")

llm = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model=GROQ_MODEL,
    temperature=0.3,
    streaming=True,
    max_retries=3,
    request_timeout=30,
)

llm_sync = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model=GROQ_MODEL,
    temperature=0,
    streaming=False,
    max_retries=3,
    request_timeout=30,
)

# =====================================================
# QUERY ANALYSIS
# =====================================================

def _pre_extract(raw: str) -> dict:
    import re
    q = raw.lower().strip()

    level = None
    level_patterns = [
        (r'\b100\s*[-\s]?l(vl|evel)?\b', "100"),
        (r'\b200\s*[-\s]?l(vl|evel)?\b', "200"),
        (r'\b300\s*[-\s]?l(vl|evel)?\b', "300"),
        (r'\b400\s*[-\s]?l(vl|evel)?\b', "400"),
        (r'\b500\s*[-\s]?l(vl|evel)?\b', "500"),
        (r'\bfirst[- ]year\b', "100"),
        (r'\bfreshm[ae]n\b', "100"),
        (r'\bpart\s*one\b', "100"),
        (r'\bpart\s*two\b', "200"),
        (r'\bpart\s*three\b', "300"),
        (r'\bpart\s*four\b', "400"),
        (r'\bpart\s*five\b', "500"),
    ]
    for pattern, val in level_patterns:
        if re.search(pattern, q):
            level = val
            break

    semester = None
    if re.search(r'\bfirst\s+sem(ester)?\b|1st\s+sem(ester)?\b', q):
        semester = "first"
    elif re.search(r'\bsecond\s+sem(ester)?\b|2nd\s+sem(ester)?\b', q):
        semester = "second"

    course_code = None
    for m in re.finditer(r'\b([A-Za-z]{2,5})[\s\-]*(\d{3}[A-Za-z]?)\b', raw):
        candidate = (m.group(1) + m.group(2)).upper()
        if get_single_course(candidate):
            course_code = candidate
            break

    option = None
    if re.search(r'\b(ai|artificial intelligence)\b', q):
        option = "AI"
    elif re.search(r'\b(network(ing)?|computer systems?)\b', q):
        option = "Computer Systems & Network"
    elif re.search(r'\b(control|intelligent systems?|robotics)\b', q):
        option = "Control & Intelligent Systems"

    is_greeting = bool(re.match(
        r'^(hi+|hello+|hey+|good\s+(morning|afternoon|evening)|howdy|sup|what.?s up)[!?,.\s]*$', q
    ))

    return {
        "level": level,
        "semester": semester,
        "course_code": course_code,
        "option": option,
        "is_greeting": is_greeting,
    }


def analyze_query(raw_query: str) -> dict:
    pre = _pre_extract(raw_query)
    q_lower = raw_query.lower().strip()

    # Location keywords
    is_location_query = bool(re.search(
        r'\b(where is|where are|location|classrooms?|venue|rooms?|lecture rooms?|theatre)\b', q_lower
    ))

    # Office keywords
    is_office_query = bool(re.search(
        r'\b(dr|prof|professor|lecturer|hod|engr)\b|office of', q_lower
    ))

    if pre["is_greeting"]:
        return {"intent": "greeting", "level": None, "semester": None, "option": None,
                "course_code": None, "room_query": None, "person_query": None, "semantic_query": raw_query}

    # Priority Order
    if is_office_query:
        return {
            "intent": "find_office",
            "level": pre["level"],
            "semester": pre["semester"],
            "option": pre.get("option"),
            "course_code": pre["course_code"],
            "room_query": None,
            "person_query": raw_query,
            "semantic_query": raw_query,
        }
    if is_electives_query(raw_query):
        return {
            "intent": "list_electives",
            "level": "500",
            "semester": None,
            "option": pre.get("option"),
            "course_code": None,
            "room_query": None,
            "person_query": None,
            "semantic_query": raw_query,
        }

    if pre["level"] and is_location_query:
        return {
            "intent": "find_room",
            "level": pre["level"],
            "semester": pre["semester"],
            "option": pre.get("option"),
            "course_code": None,
            "room_query": f"{pre['level']} level classroom",
            "person_query": None,
            "semantic_query": f"{pre['level']} level lecture room",
        }

    if pre["course_code"]:
        return {
            "intent": "find_course",
            "level": pre["level"],
            "semester": pre["semester"],
            "option": pre.get("option"),
            "course_code": pre["course_code"],
            "room_query": None, 
            "person_query": None,
            "semantic_query": raw_query,
        }

    if pre["level"] and not is_location_query:
        return {
            "intent": "list_courses",
            "level": pre["level"],
            "semester": pre["semester"],
            "option": pre.get("option"),
            "course_code": pre["course_code"],
            "room_query": None, 
            "person_query": None,
            "semantic_query": f"{pre['level']} level courses",
        }

    # LLM Fallback
    prompt = f"""Query: "{raw_query}"
Return ONLY valid JSON:
{{
  "intent": "find_room | find_office | list_courses | find_course | general_info",
  "room_query": null,
  "person_query": null,
  "semantic_query": "{raw_query}"
}}"""

    try:
        result = llm_sync.invoke([SystemMessage(content="Return only valid JSON."), HumanMessage(content=prompt)])
        raw_text = result.content.strip()
        if raw_text.startswith("```"):
            raw_text = raw_text.split("```")[1].replace("json","").strip()
        llm_out = json.loads(raw_text)
    except:
        llm_out = {"intent": "general_info", "room_query": None, "person_query": None, "semantic_query": raw_query}

    return {
        "intent": llm_out.get("intent", "general_info"),
        "level": pre["level"],
        "semester": pre["semester"],
        "option": pre.get("option"),
        "course_code": pre["course_code"],
        "room_query": llm_out.get("room_query"),
        "person_query": llm_out.get("person_query"),
        "semantic_query": llm_out.get("semantic_query", raw_query),
    }
# =====================================================
# STRUCTURED RETRIEVERS (Fixed)
# =====================================================

def get_courses(level: str, semester: str | None, option: str | None = None) -> list[dict]:
    level = str(level).strip()

    # Collect matching curriculum entries
    matches = []
    for (lvl, sem, opt), data in curriculum_db.items():
        if lvl != level:
            continue
        if sem == "electives":
            continue
        if semester and sem != semester.lower():
            continue
        if option and opt and str(opt).lower() != str(option).lower():
            continue
        matches.append((sem, opt, data))

    results = []
    seen_ids = set()
    for sem, opt, data in matches:
        # The same file can be registered under several options: return it once
        if id(data) in seen_ids:
            continue
        seen_ids.add(id(data))

        # Is this data object shared across several keys?
        sharing = [o for (_l, _s, o), d in curriculum_db.items() if d is data]
        if len(sharing) > 1:
            if option:
                label = next(
                    (o for o in sharing if o and str(o).lower() == str(option).lower()),
                    option,
                )
            else:
                label = "Common to all options"
        else:
            label = opt

        for course in data.get("courses", []):
            results.append({
                **course,
                "level":    level,
                "semester": sem,
                "option":   label,
                "total_credit_units": data.get("total_credit_units"),
            })
    return results


def get_electives(level: str = "500") -> list[dict]:
    key = (level, "electives", "All")
    data = curriculum_db.get(key, {})
    return data.get("courses", [])

def build_electives_context(level: str = "500") -> str:
    data = curriculum_db.get((level, "electives", "All"), {})
    courses = data.get("courses", [])
    if not courses:
        return f"No elective courses found for {level} level."

    lines = [f"=== {level} Level — Restricted Electives ==="]
    note = data.get("note")
    if note:
        lines.append(f"NOTE: {note}")
    lines.append("")
    lines.append(f"{'Code':<12} {'Title':<45} {'Units':<6} {'Status':<14} {'Prerequisite'}")
    lines.append("-" * 95)
    for c in courses:
        lines.append(
            f"{c.get('code',''):<12} {c.get('title',''):<45} "
            f"{c.get('unit',''):<6} {c.get('status',''):<14} {c.get('prerequisite','-')}"
        )
    return "\n".join(lines)


def is_electives_query(text: str) -> bool:
    return bool(re.search(r"\belectives?\b", text.lower()))


def get_single_course(code: str) -> dict | None:
    def norm(s):
        return re.sub(r"[\s\-]", "", str(s)).upper()

    target = norm(code)
    for (lvl, sem, opt), data in curriculum_db.items():
        for course in data.get("courses", []):
            if norm(course.get("code", "")) == target:
                return {**course, "level": lvl, "semester": sem, "option": opt}
    return None

def search_rooms(query: str) -> list[dict]:
    if not query:
        return []
    
    q = query.lower().strip()
    results = []
    
    for room in classrooms_db:
        name = str(room.get("name", "") or "").lower()
        code = str(room.get("code", "") or "").lower()
        category = str(room.get("category", "") or "").lower()
        
        searchable = f"{name} {code} {category}".lower()
        
        score = 0
        # Direct keyword matching
        if "500" in q and "500l" in name:
            score += 10
        elif "400" in q and "400l" in name:
            score += 10
        elif "300" in q and "300l" in name:
            score += 10
        elif "200" in q and "200l" in name:
            score += 10
        elif "100" in q and "100l" in name:
            score += 10
        
        # General matching
        query_words = q.split()
        for word in query_words:
            if word in searchable and len(word) > 2:
                score += 2
        
        if score >= 5:
            results.append(room)
    
    return results


def search_offices(query: str) -> list[dict]:
    if not query:
        return []
    
    q = query.lower().strip()
    # Clean common titles
    q = q.replace("dr.", "dr").replace("prof.", "prof").replace("dr ", "dr")
    
    results = []
    
    for office in offices_db:
        name = str(office.get("name", "") or "").lower()
        code = str(office.get("code", "") or "").lower()
        role = str(office.get("role", "") or "").lower()
        
        searchable = f"{name} {code} {role} {office.get('category','')} {office.get('floor','')} {office.get('wing','')}".lower()
        
        # Flexible matching logic
        score = 0
        query_words = [word for word in q.split() if len(word) > 2]
        
        for word in query_words:
            if word in name:
                score += 3          # Heavy weight for name match
            elif word in searchable:
                score += 1
        
        # Special handling for hyphenated names like "Ore-Ajayi"
        if "ajayi" in q and "ajayi" in name:
            score += 5
        
        if score >= 3:
            results.append(office)
    
    # Remove duplicates and return
    seen = set()
    unique_results = []
    for r in results:
        key = r.get("code") or r.get("name")
        if key and key not in seen:
            seen.add(key)
            unique_results.append(r)
    
    return unique_results


# =====================================================
# CONTEXT BUILDERS
# =====================================================

def build_courses_context(level: str, semester: str | None, option: str | None = None) -> str:
    courses = get_courses(level, semester, option)
    if not courses:
        return f"No courses found for {level} level{(' ' + semester + ' semester') if semester else ''}."

    is_option_level = level in ("400", "500")
    lines = []

    if not is_option_level:
        # 100-300 Level
        by_sem: dict = {}
        for c in courses:
            by_sem.setdefault(c["semester"], []).append(c)
        
        for sem in sorted(by_sem.keys()):
            sem_courses = by_sem[sem]
            total = sem_courses[0].get("total_credit_units", "?")
            lines.append(f"=== {level} Level — {sem.title()} Semester (Total: {total} units) ===")
            lines.append(f"{'Code':<12} {'Title':<45} {'Units':<6} {'Status':<20} {'Prerequisite'}")
            lines.append("-" * 100)
            for c in sem_courses:
                lines.append(
                    f"{c.get('code',''):<12} {c.get('title',''):<45} "
                    f"{c.get('unit',''):<6} {c.get('status',''):<20} {c.get('prerequisite','-')}"
                )
            lines.append("")
    else:
        # 400-500 Level with options
        from collections import defaultdict
        by_opt_sem = defaultdict(lambda: defaultdict(list))
        for c in courses:
            opt = c.get("option") or "Common"
            by_opt_sem[opt][c["semester"]].append(c)

        for opt in sorted(by_opt_sem.keys()):
            if opt == "All":
                continue
            lines.append("=" * 60)
            lines.append(f"  OPTION: {opt}")
            lines.append("=" * 60)
            for sem in sorted(by_opt_sem[opt].keys()):
                sem_courses = by_opt_sem[opt][sem]
                total = sem_courses[0].get("total_credit_units", "?")
                lines.append(f"  --- {sem.title()} Semester (Total: {total} units) ---")
                lines.append(f"  {'Code':<12} {'Title':<45} {'Units':<6} {'Status':<20} {'Prerequisite'}")
                lines.append("  " + "-" * 96)
                for c in sem_courses:
                    lines.append(
                        f"  {c.get('code',''):<12} {c.get('title',''):<45} "
                        f"{c.get('unit',''):<6} {c.get('status',''):<20} {c.get('prerequisite','-')}"
                    )
                lines.append("")

    return "\n".join(lines)


def build_room_context(query: str) -> str:
    rooms = search_rooms(query)
    
    if rooms:
        lines = ["**Classroom / Lecture Room Found:**\n"]
        for r in rooms:
            full_loc = r.get("full_location") or f"{r.get('floor','')} - {r.get('wing','')}"
            lines.append(f"**{r.get('name')}**")
            lines.append(f"Code: `{r.get('code')}`")
            lines.append(f"Floor: {r.get('floor')}")
            lines.append(f"Wing: {r.get('wing')}")
            lines.append(f"**Location:** {full_loc}")
            lines.append("")
        return "\n".join(lines)
    
    # Fallback message
    return """I couldn't find a specific room matching your query.

**Tip:** Try asking like this:
- "Where is 500L Lecture Room"
- "CO-B14 location"
- "500 level classroom"
"""


def build_office_context(query: str) -> str:
    offices = search_offices(query)
    if not offices:
        return f"❌ No office information found for **'{query}'**.\n\nYou may try searching with the full name (e.g., Dr. Ore-Ajayi)."

    lines = ["**Office Found:**\n"]
    for o in offices:
        full_loc = o.get("full_location") or f"{o.get('floor','')} — {o.get('wing','')}".strip()
        
        lines.append(f"**{o.get('name')}**")
        lines.append(f"Code: `{o.get('code')}`")
        if o.get("role"):
            lines.append(f"Role: {o.get('role')}")
        if full_loc:
            lines.append(f"**Location:** {full_loc}")
        lines.append("")

    return "\n".join(lines)


def build_course_context(code: str) -> str:
    course = get_single_course(code)
    if not course:
        return f"Course **{code}** not found in the curriculum database."
    
    opt_str = f"\nOption     : {course.get('option')}" if course.get("option") and course.get("option") not in ("All", None) else ""
    return f"""
**Course Details**
- **Code**          : {course.get('code')}
- **Title**         : {course.get('title')}
- **Units**         : {course.get('unit')}
- **Status**        : {course.get('status')}
- **Prerequisite**  : {course.get('prerequisite', '-')}
- **Level**         : {course.get('level')} Level
- **Semester**      : {str(course.get('semester','')).title()} Semester{opt_str}
""".strip()


def build_history_text() -> str:
    if not st.session_state.chat_history:
        return ""
    return "\n\n".join([
        f"User: {h['user']}\nAssistant: {h['assistant']}"
        for h in st.session_state.chat_history[-4:]
    ])

# =====================================================
# DISPLAY CHAT HISTORY
# =====================================================
for msg in st.session_state.chat_history:
    with st.chat_message("user"):
        st.markdown(msg["user"])
    with st.chat_message("assistant"):
        st.markdown(msg["assistant"])


# =====================================================
# MAIN CHAT LOOP
# =====================================================
NOT_FOUND_MARKERS = (
    "not found",
    "no courses found",
    "no office information found",
    "couldn't find a specific room",
)

def _is_empty_context(ctx: str) -> bool:
    c = (ctx or "").strip().lower()
    return (not c) or any(m in c for m in NOT_FOUND_MARKERS)

from langchain_core.messages import AIMessageChunk

def safe_stream(messages):
    try:
        for chunk in llm.stream(messages):
            yield chunk
    except Exception as e:
        print(f"[LLM error] {type(e).__name__}: {e}")
        yield AIMessageChunk(
            content="⚠️ I couldn't reach the AI service just now. Please try again in a moment."
        )


question = st.chat_input("🔍 Ask anything about the Computer Engineering Department...")

if question:
    with st.chat_message("user"):
        st.markdown(question)

    with st.spinner("Thinking..."):
        qi = analyze_query(question)

    intent       = qi.get("intent", "general_info")
    level        = qi.get("level")
    semester     = qi.get("semester")
    option       = qi.get("option")
    course_code  = qi.get("course_code")
    room_query   = qi.get("room_query")
    person_query = qi.get("person_query")
    semantic_q   = qi.get("semantic_query", question)

    if intent == "greeting":
        reply = (
            "Hello! 👋 I'm **AskDept**, your Computer Engineering department assistant at ABU Zaria.\n\n"
            "You can ask me about courses, lecturers' offices, labs, or department information."
        )
        with st.chat_message("assistant"):
            st.markdown(reply)
        st.session_state.chat_history.append({"user": question, "assistant": reply})

    else:
        context = ""
        strategy = "general_info (vector)"

        if intent == "list_courses" and level:
            context = build_courses_context(level, semester, option)
            strategy = f"structured → curriculum ({level}L)"

        elif intent == "find_course" and course_code:
            context = build_course_context(course_code)
            strategy = f"structured → single course ({course_code})"

        elif intent == "find_room":
            q = room_query or semantic_q
            context = build_room_context(q)
            strategy = f"structured → rooms ('{q}')"

        elif intent == "list_electives":
            context = build_electives_context("500")
            strategy = "structured → electives"

        elif intent == "find_office":
            q = person_query or semantic_q
            context = build_office_context(q)
            room_ctx = build_room_context(q)
            if not _is_empty_context(room_ctx):
                context += "\n\n" + room_ctx
            strategy = f"structured → offices ('{q}')"

        # Fallback to vector search
        if _is_empty_context(context) or intent in ("general_info", "other"):
            vector_docs = retriever.invoke(semantic_q)
            vec_context = "\n\n".join(d.page_content for d in vector_docs)

            if not _is_empty_context(context):
                context = context + "\n\n--- Additional information from handbook ---\n\n" + vec_context
            else:
                context = vec_context
            strategy = "vector search"

        history_text = build_history_text()

        system_prompt = f"""You are AskDept, a friendly and knowledgeable assistant for the Computer Engineering Department at Ahmadu Bello University (ABU) Zaria.

Answer using ONLY the provided context. Be clear, helpful and direct.

Guidelines:
- Never make up information. Use only facts that appear in the context.
- Answer only what was asked. Do not add extra courses, offices, rooms or topics the user did not ask about.
- If the answer is not in the context, say clearly that you don't have that information, then stop. Do not offer unrelated alternatives from the context. You may suggest one short way to rephrase, or tell them to contact the department office.
- When the context lists several options, never merge them into a single table without saying which option each course belongs to. Either add an "Option(s)" column or show a separate table per option, and say which courses are common to all options.
- If the user asks about a specific course code, lecturer or room that does not appear in the context, say it is not in your records. Do not substitute a different one.
- For yes/no questions, begin with the exact distinction the data supports. For electives: no single elective is compulsory, but students must take the required number of units from the list.
- Use markdown tables for course lists. For single facts (a credit total, an office location) answer in one or two sentences.
- Keep answers concise and friendly.
Chat History:
{history_text}

--- Context ---
{context}
--- End of Context ---"""

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=question),
        ]

        with st.chat_message("assistant"):
            placeholder = st.empty()
            answer_text = ""
            for chunk in safe_stream(messages):
                if chunk.content:
                    answer_text += chunk.content
                    placeholder.markdown(answer_text)

        st.session_state.chat_history.append({"user": question, "assistant": answer_text})
        st.session_state.eval_log.append({
            "question": question,
            "intent": intent,
            "strategy": strategy,
            "answer_length": len(answer_text),
        })


# =====================================================
# SIDEBAR
# =====================================================
st.sidebar.header("⚙️ Controls")

if st.sidebar.button("🧹 Clear Chat"):
    st.session_state.chat_history = []
    st.rerun()

if st.sidebar.button("♻️ Rebuild Vector Database"):
    import shutil
    if os.path.exists(PERSIST_DIR):
        shutil.rmtree(PERSIST_DIR)
    st.cache_resource.clear()
    st.sidebar.success("Vector DB cleared. Refresh the page.")

st.sidebar.markdown("---")
st.sidebar.subheader("📊 Database Info")
st.sidebar.write(f"**Curriculum files:** {len(curriculum_db)}")
st.sidebar.write(f"**Total courses:** {sum(len(v.get('courses', [])) for v in curriculum_db.values())}")
st.sidebar.write(f"**Classrooms:** {len(classrooms_db)}")
st.sidebar.write(f"**Offices:** {len(offices_db)}")

st.sidebar.markdown("---")
if st.session_state.eval_log:
    last = st.session_state.eval_log[-1]
    st.sidebar.caption(f"Last Intent: **{last['intent']}**")