import os
import json
import streamlit as st
from dotenv import load_dotenv

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

# =====================================================
# ENV SETUP
# =====================================================
load_dotenv()

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# =====================================================
# STREAMLIT CONFIG
# =====================================================
st.set_page_config(page_title="AskDept | Computer Engineering", page_icon="🏫")
st.title("🏫 AskDept | Computer Engineering Department")



# =====================================================
# SESSION STATE
# =====================================================
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "eval_log" not in st.session_state:
    st.session_state.eval_log = []

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
def load_curriculum() -> dict:
    """
    Keys: (level, semester, option)
    - 100-300L: option=None  e.g. ("100", "first", None)
    - 400-500L options:      e.g. ("400", "first", "AI")
    - 500L electives:        ("500", "electives", "All")
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
                    docs   = loader.load()
                    for d in docs:
                        d.metadata["source"] = fname
                        d.metadata["type"]   = "pdf"
                    all_docs.extend(docs)
                except Exception as e:
                    st.warning(f"Could not load {fname}: {e}")
    if not all_docs:
        from langchain_core.documents import Document
        all_docs = [Document(page_content="placeholder", metadata={})]
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits   = splitter.split_documents(all_docs)
    db       = FAISS.from_documents(splits, embeddings)
    os.makedirs(PERSIST_DIR, exist_ok=True)
    db.save_local(PERSIST_DIR)
    return db

vector_db = load_or_create_vectorstore()
retriever = vector_db.as_retriever(
    search_type="mmr", search_kwargs={"k": 8, "fetch_k": 20}
)

# =====================================================
# LLM
# =====================================================

llm = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model="llama-3.1-8b-instant",
    temperature=0.3,
    streaming=True,
)
llm_sync = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model="llama-3.1-8b-instant",
    temperature=0,
    streaming=False,
)

# =====================================================
# QUERY ANALYSIS
# =====================================================

def _pre_extract(raw: str) -> dict:
    import re
    q = raw.lower().strip()

    # ── Level ──────────────────────────────────────────
    level = None
    level_patterns = [
        (r'\b100\s*[-\s]?l(vl|evel)?\b',  "100"),
        (r'\b200\s*[-\s]?l(vl|evel)?\b',  "200"),
        (r'\b300\s*[-\s]?l(vl|evel)?\b',  "300"),
        (r'\b400\s*[-\s]?l(vl|evel)?\b',  "400"),
        (r'\b500\s*[-\s]?l(vl|evel)?\b',  "500"),
        (r'\bfirst[- ]year\b',             "100"),
        (r'\bfreshm[ae]n\b',               "100"),
        (r'\bpart\s*one\b',                "100"),
        (r'\bpart\s*two\b',                "200"),
        (r'\bpart\s*three\b',              "300"),
        (r'\bpart\s*four\b',               "400"),
        (r'\bpart\s*five\b',               "500"),
    ]
    for pattern, val in level_patterns:
        if re.search(pattern, q):
            level = val
            break

    # ── Semester ───────────────────────────────────────
    semester = None
    if re.search(r'\bfirst\s+sem(ester)?\b|1st\s+sem(ester)?\b', q):
        semester = "first"
    elif re.search(r'\bsecond\s+sem(ester)?\b|2nd\s+sem(ester)?\b', q):
        semester = "second"

    # ── Course code ────────────────────────────────────
    course_code = None
    cc_match = re.search(r'\b([A-Za-z]{2,5}\s*\d{3}[A-Za-z]?)\b', raw)
    if cc_match:
        course_code = cc_match.group(1).replace(" ", "").upper()

    # ── Option ─────────────────────────────────────────
    option = None
    if re.search(r'\b(ai|artificial intelligence)\b', q):
        option = "AI"
    elif re.search(r'\b(network(ing)?|computer systems?)\b', q):
        option = "Computer Systems & Network"
    elif re.search(r'\b(control|intelligent systems?|robotics)\b', q):
        option = "Control & Intelligent Systems"

    # ── Greeting ───────────────────────────────────────
    is_greeting = bool(re.match(
        r'^(hi+|hello+|hey+|good\s+(morning|afternoon|evening)|howdy|sup|what.?s up)[!?,.\s]*$', q
    ))

    return {
        "level":        level,
        "semester":     semester,
        "course_code":  course_code,
        "option":       option,
        "is_greeting":  is_greeting,
    }


def analyze_query(raw_query: str) -> dict:
    pre = _pre_extract(raw_query)

    if pre["is_greeting"]:
        return {
            "intent": "greeting",
            "level": None, "semester": None, "option": None,
            "course_code": None, "room_query": None, "person_query": None,
            "semantic_query": raw_query,
        }

    if pre["course_code"] and not pre["level"]:
        return {
            "intent": "find_course",
            "level": None, "semester": None, "option": None,
            "course_code": pre["course_code"],
            "room_query": None, "person_query": None,
            "semantic_query": raw_query,
        }

    room_keywords   = ["lab", "room", "classroom", "library", "workshop",
                       "studio", "hall", "office", "hub", "academy", "floor", "wing"]
    office_keywords = ["dr.", "dr ", "prof.", "prof ", "professor", "lecturer",
                       "hod", "head of department", "secretary", "technologist",
                       "exam office", "registration", "coesa", "adjunct"]
    q_lower = raw_query.lower()
    has_room_kw   = any(k in q_lower for k in room_keywords)
    has_office_kw = any(k in q_lower for k in office_keywords)

    if pre["level"] and not has_room_kw and not has_office_kw:
        opt = pre.get("option")
        sem_label = f" {pre['semester']} semester" if pre["semester"] else ""
        opt_label = f" {opt} option" if opt else ""
        return {
            "intent": "list_courses",
            "level": pre["level"],
            "semester": pre["semester"],
            "option": opt,
            "course_code": pre["course_code"],
            "room_query": None, "person_query": None,
            "semantic_query": f"{pre['level']} level{sem_label}{opt_label} courses",
        }

    known = []
    if pre["level"]:       known.append(f'level is already extracted as "{pre["level"]}"')
    if pre["semester"]:    known.append(f'semester is already extracted as "{pre["semester"]}"')
    if pre["course_code"]: known.append(f'course_code is already extracted as "{pre["course_code"]}"')
    known_note = ("Note — already extracted by system: " + "; ".join(known)) if known else ""

    prompt = f"""You are a query analyzer for a Nigerian university Computer Engineering department chatbot.
Return ONLY a valid JSON object — no markdown, no preamble.

Query: "{raw_query}"
{known_note}

Return exactly:
{{
  "intent": "find_room | find_office | list_courses | find_course | general_info | other",
  "room_query":   "exact room/lab/place name from query, or null",
  "person_query": "exact person name or role from query e.g. 'Dr. Yahaya', 'HOD', or null",
  "semantic_query": "2-5 keyword phrase for vector search"
}}

Intent rules:
- Room, lab, classroom, library, workshop, lecture room, floor → find_room
- Dr., Prof., lecturer, HOD, secretary, office, technologist, registration → find_office
- Courses, subjects, curriculum, units → list_courses
- Specific course code like COEN301 → find_course
- Anything else (regulations, requirements, policies) → general_info"""

    messages = [
        SystemMessage(content="You are a query analysis assistant. Return only valid JSON."),
        HumanMessage(content=prompt),
    ]
    try:
        result  = llm_sync.invoke(messages)
        raw     = result.content.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        llm_out = json.loads(raw.strip())
    except Exception:
        llm_out = {
            "intent": "general_info",
            "room_query": None, "person_query": None,
            "semantic_query": raw_query,
        }

    return {
        "intent":        llm_out.get("intent", "general_info"),
        "level":         pre["level"],
        "semester":      pre["semester"],
        "option":        pre.get("option"),
        "course_code":   pre["course_code"] or llm_out.get("course_code"),
        "room_query":    llm_out.get("room_query"),
        "person_query":  llm_out.get("person_query"),
        "semantic_query": llm_out.get("semantic_query", raw_query),
    }

# =====================================================
# STRUCTURED RETRIEVERS
# =====================================================

def get_courses(level: str, semester: str | None, option: str | None = None) -> list[dict]:
    results = []
    level = str(level).strip()
    for (lvl, sem, opt), data in curriculum_db.items():
        if lvl != level:
            continue
        if sem == "electives":
            continue
        if semester and sem != semester.lower():
            continue
        if option and opt and opt.lower() != option.lower():
            continue
        for course in data.get("courses", []):
            results.append({
                **course,
                "level":    lvl,
                "semester": sem,
                "option":   opt,
                "total_credit_units": data.get("total_credit_units"),
            })
    return results


def get_electives(level: str = "500") -> list[dict]:
    key  = (level, "electives", "All")
    data = curriculum_db.get(key, {})
    return data.get("courses", [])


def get_single_course(code: str) -> dict | None:
    code = code.strip().upper()
    for (lvl, sem, opt), data in curriculum_db.items():
        for course in data.get("courses", []):
            if course.get("code", "").upper() == code:
                return {**course, "level": lvl, "semester": sem, "option": opt}
    return None


def search_rooms(query: str) -> list[dict]:
    q = query.lower()
    results = []
    for room in classrooms_db:
        searchable = " ".join([
            room.get("name") or "",
            room.get("category") or "",
            room.get("code") or "",
            room.get("floor") or "",
            room.get("wing") or "",
        ]).lower()
        if any(word in searchable for word in q.split() if len(word) > 2):
            results.append(room)
    return results


def search_offices(query: str) -> list[dict]:
    q = query.lower()
    results = []
    for office in offices_db:
        searchable = " ".join([
            office.get("name") or "",
            office.get("code") or "",
            office.get("floor") or "",
            office.get("wing") or "",
            office.get("category") or "",
        ]).lower()
        if any(word in searchable for word in q.split() if len(word) > 2):
            results.append(office)
    return results

# =====================================================
# CONTEXT BUILDERS
# =====================================================

def build_courses_context(level: str, semester: str | None, option: str | None = None) -> str:
    courses = get_courses(level, semester, option)
    if not courses:
        return f"No courses found for {level} level{(' ' + semester + ' semester') if semester else ''}."

    is_option_level = level in ("400", "500")

    if not is_option_level:
        by_sem: dict[str, list] = {}
        for c in courses:
            by_sem.setdefault(c["semester"], []).append(c)
        lines = []
        for sem in sorted(by_sem.keys()):
            sem_courses = by_sem[sem]
            total = curriculum_db.get((level, sem, None), {}).get("total_credit_units", "?")
            lines.append(f"=== {level} Level — {sem.title()} Semester (Total: {total} units) ===")
            lines.append(f"{'Code':<12} {'Title':<45} {'Units':<6} {'Status':<20} {'Prerequisite'}")
            lines.append("-" * 100)
            for c in sem_courses:
                lines.append(
                    f"{c.get('code',''):<12} {c.get('title',''):<45} "
                    f"{c.get('unit',''):<6} {c.get('status',''):<20} {c.get('prerequisite','-')}"
                )
            lines.append("")
        return "\n".join(lines)

    else:
        from collections import defaultdict
        by_opt_sem: dict = defaultdict(lambda: defaultdict(list))
        for c in courses:
            opt = c.get("option") or "Common"
            by_opt_sem[opt][c["semester"]].append(c)

        lines = []
        if level == "400":
            lines.append("NOTE: All 400 Level students do SIWES (6 units) in Second Semester regardless of option.\n")

        for opt in sorted(by_opt_sem.keys()):
            if opt == "All":
                continue
            lines.append("=" * 60)
            lines.append(f"  OPTION: {opt}")
            lines.append("=" * 60)
            for sem in sorted(by_opt_sem[opt].keys()):
                sem_courses = by_opt_sem[opt][sem]
                total = curriculum_db.get((level, sem, opt), {}).get("total_credit_units", "?")
                lines.append(f"  --- {sem.title()} Semester (Total: {total} units) ---")
                lines.append(f"  {'Code':<12} {'Title':<45} {'Units':<6} {'Status':<20} {'Prerequisite'}")
                lines.append("  " + "-" * 96)
                for c in sem_courses:
                    lines.append(
                        f"  {c.get('code',''):<12} {c.get('title',''):<45} "
                        f"{c.get('unit',''):<6} {c.get('status',''):<20} {c.get('prerequisite','-')}"
                    )
                lines.append("")

        if level == "500":
            electives = get_electives("500")
            if electives:
                lines.append("=" * 60)
                lines.append("  RESTRICTED ELECTIVES (choose 4 units from the following)")
                lines.append("=" * 60)
                lines.append(f"  {'Code':<12} {'Title':<45} {'Units':<6} {'Status'}")
                lines.append("  " + "-" * 80)
                for e in electives:
                    lines.append(
                        f"  {e.get('code',''):<12} {e.get('title',''):<45} "
                        f"{e.get('unit',''):<6} {e.get('status','')}"
                    )
        return "\n".join(lines)


def build_room_context(query: str) -> str:
    rooms = search_rooms(query)
    if not rooms:
        return f"No rooms found matching '{query}'."
    lines = ["Room search results:"]
    for r in rooms:
        full_loc = r.get("full_location") or f"{r.get('floor','')} {r.get('wing','')}"
        lines.append(f"- {r['name']} (Code: {r['code']}) | {r['category']} | {full_loc}")
    return "\n".join(lines)


def build_office_context(query: str) -> str:
    offices = search_offices(query)
    if not offices:
        return f"No offices found matching '{query}'."
    lines = ["Office search results:"]
    for o in offices:
        full_loc = o.get("full_location") or f"{o.get('floor','')} {o.get('wing') or ''}"
        lines.append(f"- {o['name']} (Code: {o['code']}) | {full_loc}")
    return "\n".join(lines)


def build_course_context(code: str) -> str:
    course = get_single_course(code)
    if not course:
        return f"Course {code} not found in the curriculum database."
    opt_str = f"\n  Option     : {course['option']}" if course.get("option") and course["option"] not in ("All", None) else ""
    return (
        f"Course Details:\n"
        f"  Code       : {course.get('code')}\n"
        f"  Title      : {course.get('title')}\n"
        f"  Units      : {course.get('unit')}\n"
        f"  Status     : {course.get('status')}\n"
        f"  Prerequisite: {course.get('prerequisite', '-')}\n"
        f"  Level      : {course.get('level')} Level\n"
        f"  Semester   : {str(course.get('semester','')).title()} Semester"
        f"{opt_str}"
    )


def build_history_text() -> str:
    return "\n".join([
        f"User: {h['user']}\nAssistant: {h['assistant']}"
        for h in st.session_state.chat_history[-4:]
    ])

# =====================================================
# DISPLAY EXISTING HISTORY
# =====================================================

for msg in st.session_state.chat_history:
    with st.chat_message("user"):
        st.markdown(msg["user"])
    with st.chat_message("assistant"):
        st.markdown(msg["assistant"])


# =====================================================
# MAIN CHAT LOOP
# =====================================================

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
            "Ask me about **courses**, **office locations**, **labs**, or **department policies**.\n\n"
            "How can I help you today?"
        )
        with st.chat_message("assistant"):
            st.markdown(reply)
        st.session_state.chat_history.append({"user": question, "assistant": reply})

    else:
        context  = ""
        strategy = "general_info (vector)"

        if intent == "list_courses" and level:
            context   = build_courses_context(level, semester, option)
            opt_label = f" [{option}]" if option else ""
            strategy  = f"structured → curriculum ({level}L, {semester or 'all semesters'}{opt_label})"

        elif intent == "find_course" and course_code:
            context  = build_course_context(course_code)
            strategy = f"structured → single course ({course_code})"

        elif intent == "find_room":
            q        = room_query or semantic_q
            context  = build_room_context(q)
            strategy = f"structured → classrooms ('{q}')"

        elif intent == "find_office":
            q        = person_query or semantic_q
            context  = build_office_context(q)
            room_ctx = build_room_context(q)
            if "No rooms found" not in room_ctx:
                context += "\n\n" + room_ctx
            strategy = f"structured → offices ('{q}')"

        if not context or "not found" in context.lower() or intent in ("general_info", "other"):
            vector_docs = retriever.invoke(semantic_q)
            vec_context = "\n\n".join(d.page_content for d in vector_docs)
            if context and "not found" not in context.lower():
                context = context + "\n\n--- Additional context from handbook ---\n\n" + vec_context
            else:
                context  = vec_context
                strategy = "vector search → handbook"

        history_text = build_history_text()

        system_prompt = f"""You are AskDept, a friendly and knowledgeable assistant for the Computer Engineering Department at Ahmadu Bello University (ABU) Zaria, Nigeria.

Your job is to answer student and staff questions clearly and helpfully using ONLY the provided context.

Guidelines:
- Be warm, clear, and direct. Sound like a helpful human department guide, not a robot.
- NEVER invent information. If it is not in the context, say: "I don't have that information in my knowledge base — you may want to check with the department office."
- When listing courses, format them as a neat markdown table with columns: Code | Title | Units | Status | Prerequisite.
- When giving room or office locations, state the floor and wing clearly.
- If the user asks about multiple semesters, present First Semester first, then Second Semester, each in their own table.
- Keep answers concise but complete — don't waffle.
- CRITICAL: If the context contains course tables or structured data, you MUST present it. Never say "I don't have that information" when course data is clearly present in the context. The structured data in the context IS your knowledge base — trust it completely.

Chat History (for context):
{history_text}

--- Retrieved Context ---
{context}
--- End of Context ---"""

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=question),
        ]

        with st.chat_message("assistant"):
            placeholder = st.empty()
            answer_text = ""
            for chunk in llm.stream(messages):
                if chunk.content:
                    answer_text += chunk.content
                    placeholder.markdown(answer_text)

        st.session_state.chat_history.append({"user": question, "assistant": answer_text})
        st.session_state.eval_log.append({
            "question":      question,
            "intent":        intent,
            "strategy":      strategy,
            "answer_length": len(answer_text),
        })

        with st.expander("🔍 Query Analysis"):
            st.json({
                "intent":       intent,
                "level":        level,
                "semester":     semester,
                "option":       option,
                "course_code":  course_code,
                "room_query":   room_query,
                "person_query": person_query,
                "strategy":     strategy,
            })

        with st.expander("📚 Raw Context Sent to LLM"):
            st.text(context[:3000] + ("..." if len(context) > 3000 else ""))


# =====================================================
# SIDEBAR
# =====================================================

st.sidebar.header("⚙️ Controls")

if st.sidebar.button("🧹 Clear Chat Memory"):
    st.session_state.chat_history = []
    st.rerun()

if st.sidebar.button("♻️ Rebuild Vector Database"):
    import shutil
    if os.path.exists(PERSIST_DIR):
        shutil.rmtree(PERSIST_DIR)
    st.cache_resource.clear()
    st.sidebar.success("Vector DB cleared — will rebuild on next run.")

st.sidebar.markdown("---")
st.sidebar.subheader("📦 Data Loaded")
total_courses = sum(len(v.get("courses", [])) for v in curriculum_db.values())
st.sidebar.write(f"📖 Curriculum files: **{len(curriculum_db)}**")
st.sidebar.write(f"📚 Total courses: **{total_courses}**")
st.sidebar.write(f"🏫 Classrooms: **{len(classrooms_db)}**")
st.sidebar.write(f"🚪 Offices: **{len(offices_db)}**")

st.sidebar.markdown("---")
st.sidebar.subheader("📊 Last Query")
if st.session_state.eval_log:
    last = st.session_state.eval_log[-1]
    st.sidebar.write(f"**Intent:** {last['intent']}")
    st.sidebar.write(f"**Strategy:** {last['strategy']}")
    st.sidebar.write(f"**Answer length:** {last['answer_length']} chars")
else:
    st.sidebar.caption("No queries yet.")