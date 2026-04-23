import streamlit as st
import pdfplumber
import re
import requests
from bs4 import BeautifulSoup
import json
import csv
import pandas as pd
import time 
from types import SimpleNamespace
from functools import lru_cache

from groq import Groq


class TigerChat:
    def __init__(self, client, system_prompt):
        self.client = client
        self.model = "llama-3.3-70b-versatile"
        self.messages = [{"role": "system", "content": system_prompt}]

    def send_message(self, text):
        self.messages.append({"role": "user", "content": text})
        response = self.client.chat.completions.create(
            model=self.model,
            messages=self.messages,
            temperature=0.2,
        )
        response_text = (response.choices[0].message.content or "").strip()
        self.messages.append({"role": "assistant", "content": response_text})
        return SimpleNamespace(text=response_text)

# ─────────────────────────────────────────────────────────────────
# MIZZOU ADVANCED INTERFACE CONFIGURATION
# ─────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Mizzou Academic Advisor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

MIZZOU_GOLD = "#F1B82D"
MIZZOU_BLACK = "#000000"
MIZZOU_GREY = "#E1E1E1"

# ─────────────────────────────────────────────────────────────────
# AI ADVISOR CONFIGURATION
# ─────────────────────────────────────────────────────────────────
API_KEY = st.secrets["GROQ_API_KEY"]

# ─────────────────────────────────────────────────────────────────
# MIZZOU SIDEBAR & INTERFACE
# ─────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&display=swap');
    html, body, [class*="css"], [data-testid="stAppViewContainer"] {
        font-family: 'IBM Plex Sans', sans-serif;
    }
    .block-container {
        max-width: 1200px;
        padding-top: 1.25rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("### Mizzou Academic Advisor")
    st.info("""This AI-powered advisor helps you navigate your Mizzou degree plan by 
    parsing your unofficial transcript and dynamically scraping live catalog requirements.""")
    st.warning("⚠️ **Disclaimer:** This tool is for informational purposes. Always consult an official human academic advisor before enrolling.")
    
    if st.button("Reset Session"):
        st.session_state.clear()
        st.rerun()

st.markdown(f"""
<h1 style='color:{MIZZOU_BLACK};'>University of Missouri <span style='color:{MIZZOU_GOLD};'>Academic Advisor</span></h1>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────
# TRANSCRIPT PARSING 
# ─────────────────────────────────────────────────────────────────
def parse_transcript(file_obj):
    lines = []
    with pdfplumber.open(file_obj) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                lines.extend(text.splitlines())
    programs = _parse_program_entries(lines)
    majors = [p["name"] for p in programs if p["type"] == "major"]
    minors = [p["name"] for p in programs if p["type"] == "minor"]
    certificates = [p["name"] for p in programs if p["type"] == "certificate"]
    emphases = [p["name"] for p in programs if p["type"] == "emphasis"]
    return {
        "name":     _parse_name(lines),
        "programs": programs,
        "majors":   majors or _parse_majors(lines),
        "minors":   minors,
        "certificates": certificates,
        "emphases": emphases or _parse_emphases(lines),
        "gpa":      _parse_gpa_hours(lines)[0],
        "hours":    _parse_gpa_hours(lines)[1],
        "courses":  _parse_courses(lines),
    }

def _parse_name(lines):
    for line in lines:
        m = re.match(r"^Name:\s*(.+)", line.strip())
        if m:
            parts = m.group(1).strip().split(",", 1)
            return f"{parts[1].strip()} {parts[0].strip()}" if len(parts) == 2 else parts[0]
    return "Unknown"

SKIP = {
    "LOCAL CAMPUS CREDITS UGRD", "EXAM CREDIT",
    "MISSOURI CIVICS EXAMINATION", "MU GENERAL EDUCATION MET",
    "GOOD STANDING", "STUDENT ACADEMIC PROFILE",
}

PROGRAM_PREFIX_MAP = {
    "major": "major",
    "majors": "major",
    "minor": "minor",
    "minors": "minor",
    "certificate": "certificate",
    "certificates": "certificate",
    "emphasis": "emphasis",
    "emphases": "emphasis",
}

def _dedupe_programs(programs):
    seen = set()
    deduped = []
    for item in programs:
        key = (item["type"], _normalize_lookup_key(item["name"]))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped

def _parse_program_entries(lines):
    sem_re = re.compile(r"^(FALL|SPNG|SUM)\s+\d{4}", re.IGNORECASE)
    program_label_re = re.compile(r"^(Major|Majors|Minor|Minors|Certificate|Certificates|Emphasis|Emphases)\s*:?\s*(.+)$", re.IGNORECASE)
    programs = []

    for raw_line in lines:
        s = _normalize_text(raw_line)
        if not s:
            continue
        s_upper = s.upper()
        if s_upper in SKIP:
            continue
        if "UGRD CUM:" in s_upper or sem_re.match(s_upper):
            break

        match = program_label_re.match(s)
        if match:
            ptype = PROGRAM_PREFIX_MAP.get(match.group(1).lower(), "")
            value = _normalize_text(match.group(2))
            if ptype and value:
                programs.append({"type": ptype, "name": value})
            continue

        inline_patterns = [
            ("minor", r"\bMinor in\s+(.+)$"),
            ("certificate", r"\b(?:Undergraduate|Graduate)?\s*Certificate in\s+(.+)$"),
            ("emphasis", r"\bEmphasis in\s+(.+)$"),
        ]
        for ptype, pattern in inline_patterns:
            inline_match = re.search(pattern, s, re.IGNORECASE)
            if inline_match:
                programs.append({"type": ptype, "name": _normalize_text(inline_match.group(1))})
                break

    if not any(p["type"] == "major" for p in programs):
        for major in _parse_majors(lines):
            programs.append({"type": "major", "name": major})

    return _dedupe_programs(programs)

def _parse_majors(lines):
    sem_re    = re.compile(r"^(FALL|SPNG|SUM)\s+\d{4}", re.IGNORECASE)
    course_re = re.compile(r"^[A-Za-z][A-Za-z0-9_]+\s+\d+\w*\s+.+\s+(CR|IP|[A-D][+-]?|W|F)\s+\d+\.\d+")
    gpa_re    = re.compile(r"UGRD (Term|CUM):")
    indices   = [i for i, l in enumerate(lines) if sem_re.match(l.strip())]
    if not indices:
        return []
    majors = []
    for line in lines[indices[-1] + 1:]:
        s = line.strip()
        if not s:
            continue
        if gpa_re.search(s) or course_re.match(s):
            break
        if s.upper() in SKIP:
            continue
        if re.search(r"[A-Za-z]{3,}", s) and not re.search(r"\d{3,}", s):
            if s not in majors:
                majors.append(s)
    return majors

def _parse_emphases(lines):
    emphases = []
    for line in lines:
        s = line.strip().upper()
        # Stop looking once we hit grades or semester headers to avoid matching course titles
        if "UGRD" in s or "TERM:" in s or "COURSE" in s or re.match(r"^(FALL|SPNG|SUM)\s+\d{4}", s):
            break
        if "FINANCE" in s: emphases.append("FINANCE")
        if "MANAGEMENT" in s: emphases.append("MANAGEMENT")
        if "MARKETING" in s: emphases.append("MARKETING")
        if "ACCOUNTANCY" in s or "ACCOUNTING" in s: emphases.append("ACCOUNTANCY")
        if "STATISTICS" in s: emphases.append("STATISTICS")
        if "MATHEMATICS" in s: emphases.append("MATHEMATICS")
        if "COMPUTER SCIENCE" in s: emphases.append("COMPUTER SCIENCE")
    return list(set(emphases))

def _parse_gpa_hours(lines):
    gpa = hrs = None
    pat = re.compile(r"UGRD CUM:\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)")
    for line in lines:
        m = pat.search(line)
        if m:
            hrs = float(m.group(2))
            gpa = float(m.group(4))
    return gpa, hrs

COURSE_RE = re.compile(
    r"^([A-Za-z][A-Za-z0-9_\s]{1,12}?)\s+"
    r"(\d+\w*)\s+"
    r"(.+?)\s+"
    r"(CR|IP|[A-D][+-]?|W|F)\s+"
    r"(\d+\.\d+)"
)

def _parse_courses(lines):
    courses, seen = [], set()
    for line in lines:
        m = COURSE_RE.match(line.strip())
        if m:
            prefix = m.group(1).strip().upper().replace("_", " ")
            number = m.group(2).strip().upper()
            code   = f"{prefix} {number}"
            if code not in seen:
                seen.add(code)
                grade = m.group(4).strip()
                courses.append({
                    "code":    code,
                    "title":   m.group(3).strip(),
                    "grade":   grade,
                    "credits": float(m.group(5)),
                    "status":  ("In Progress" if grade == "IP"
                                else "Withdrawn" if grade == "W"
                                else "Complete"),
                })
    return courses

# ─────────────────────────────────────────────────────────────────
# DYNAMIC CATALOG LOOKUP & SMART SCRAPING
# ─────────────────────────────────────────────────────────────────
MAJOR_ALIASES = {
    "BUS OR ACCTCY": "BUSINESS",  
    "BUS": "BUSINESS",
    "ACCTCY": "ACCOUNTANCY",
    "COMP SCI": "COMPUTER SCIENCE",
    "MECH ENG": "MECHANICAL ENG"
}

DEGREE_INDEX_URL = "https://catalog.missouri.edu/degreesanddegreeprograms/"
ADDITIONAL_INDEX_URL = "https://catalog.missouri.edu/additionalcertificatesandminors/"

# Structured Dict to dynamically target URLs based on detected emphasis
CATALOG_URLS = {
    "ACCOUNTANCY": {
        "CORE": "https://catalog.missouri.edu/collegeofbusiness/accountancy/bsacc-accountancy/"
    },
    "DATA SCIENCE": {
        "CORE": "https://catalog.missouri.edu/collegeofengineering/datascience/bs-data-science/"
    },
    "COMPUTER SCIENCE": {
        "CORE": "https://catalog.missouri.edu/collegeofengineering/computerscience/bs-computer-science/"
    },
    "MECHANICAL ENG": {
        "CORE": "https://catalog.missouri.edu/collegeofengineering/mechanicalengineering/bs-mechanical-engineering/"
    },
    "BUSINESS": {
        "CORE": "https://catalog.missouri.edu/collegeofbusiness/businessadministration/bsba-business-administration/",
        "FINANCE": "https://catalog.missouri.edu/collegeofbusiness/businessadministration/bsba-business-administration-emphasis-finance-banking/",
        "MANAGEMENT": "https://catalog.missouri.edu/collegeofbusiness/businessadministration/bsba-business-administration-emphasis-management/",
        "MARKETING": "https://catalog.missouri.edu/collegeofbusiness/businessadministration/bsba-business-administration-emphasis-marketing/",
        "ACCOUNTANCY": "https://catalog.missouri.edu/collegeofbusiness/accountancy/bsacc-accountancy/"
    },
    "ECONOMICS": {
        "CORE": "https://catalog.missouri.edu/collegeofartsandscience/economics/bs-economics/"
    },
}

def _merge_reqs(target, source):
    for k, v in source.items():
        if k not in target:
            target[k] = v
        else:
            for item in v:
                if item[0] not in [x[0] for x in target[k]]:
                    target[k].append(item)

def _sum_required_credits(courses):
    return sum(credits for code, _, credits in courses if not code.startswith("OR "))

def _normalize_text(text):
    return re.sub(r"\s+", " ", text or "").strip()

def _normalize_lookup_key(text):
    return re.sub(r"[^A-Z0-9]+", " ", (text or "").upper()).strip()

def _name_tokens(text):
    stopwords = {"AND", "OF", "IN", "THE"}
    return [token for token in _normalize_lookup_key(text).split() if token and token not in stopwords]

def _extract_course_credits_from_row(tag, title_text=""):
    texts = [_normalize_text(td.get_text(" ", strip=True)) for td in tag.find_all("td")]
    texts.append(_normalize_text(title_text))

    for text in reversed(texts):
        if not text:
            continue
        paren_match = re.search(r"\((\d+(?:\.\d+)?)\)\s*$", text)
        if paren_match:
            return float(paren_match.group(1))

        trailing_match = re.search(r"(?<!\d)(\d+(?:\.\d+)?)\s*$", text)
        if trailing_match and len(text.split()) <= 8:
            return float(trailing_match.group(1))

    return 3.0

def _extract_program_name(page_title):
    title = _normalize_text(page_title)
    if " with Emphasis in " in title:
        return title.split(" with Emphasis in ", 1)[1]
    if " in " in title:
        return title.split(" in ", 1)[1]
    return title

def _get_page_title(soup):
    h1 = soup.find("h1")
    return _normalize_text(h1.get_text(" ", strip=True)) if h1 else ""

def _extract_business_emphasis_name(soup, section_names=None):
    page_title = _get_page_title(soup)
    if " with Emphasis in " not in page_title:
        if section_names:
            for section_name in section_names:
                match = re.match(r"^(Required|Additional)\s+(.+?)\s+Courses$", _normalize_text(section_name), re.IGNORECASE)
                if not match:
                    continue

                candidate = _normalize_text(match.group(2))
                candidate_upper = candidate.upper()
                if candidate_upper in {"CORE", "BUSINESS CORE"}:
                    continue
                return candidate
        return ""
    return _extract_program_name(page_title)

def _classify_business_emphasis_section(section_name, emphasis_name):
    section_clean = _normalize_text(section_name)
    if ": " in section_clean:
        section_clean = section_clean.split(": ", 1)[1].strip()
    section_upper = section_clean.upper()
    emphasis_upper = _normalize_text(emphasis_name).upper()

    if section_upper == "EMPHASIS SUPPORT COURSES":
        return "support"

    if not emphasis_upper:
        return ""

    match = re.match(r"^(Required|Additional)\s+(.+?)\s+Courses$", section_clean, re.IGNORECASE)
    if not match:
        match = re.match(r"^(Required|Additional)\s+(.+?)\s+Emphasis\s+Courses$", section_clean, re.IGNORECASE)
    if not match:
        return ""

    section_tokens = set(_name_tokens(match.group(2)))
    emphasis_tokens = set(_name_tokens(emphasis_name))
    overlap = len(section_tokens & emphasis_tokens)
    if not (section_tokens and emphasis_tokens and overlap > 0):
        return ""

    section_kind = match.group(1).strip().lower()
    if section_kind == "required":
        return "required"
    if section_kind == "additional":
        return "additional"
    return ""

def _is_business_emphasis_section(section_name, emphasis_name):
    return _classify_business_emphasis_section(section_name, emphasis_name) in {"required", "additional", "support"}

def _build_emphasis_phrase_pattern(emphasis_name):
    tokens = _name_tokens(emphasis_name)
    if not tokens:
        return ""
    return r"\s*(?:&|AND)?\s*".join(re.escape(token) for token in tokens)

def _best_business_emphasis_url(target_emphasis):
    target_tokens = set(_name_tokens(target_emphasis))
    if not target_tokens:
        return ""

    best_score = 0
    best_url = ""
    for key, url in CATALOG_URLS.get("BUSINESS", {}).items():
        if key == "CORE":
            continue
        try:
            resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
            if resp.status_code != 200:
                continue
            soup = BeautifulSoup(resp.text, "html.parser")
            emphasis_name = _extract_business_emphasis_name(soup)
            if not emphasis_name:
                continue
            overlap = len(set(_name_tokens(emphasis_name)) & target_tokens)
            if overlap > best_score:
                best_score = overlap
                best_url = url
        except requests.RequestException:
            continue
    return best_url

def _parse_program_label(program_label):
    if ": " not in program_label:
        return "major", program_label
    lhs, rhs = program_label.split(": ", 1)
    ptype = lhs.strip().lower()
    if ptype not in {"major", "minor", "certificate", "emphasis"}:
        ptype = "major"
    return ptype, rhs.strip()

def _normalize_focus_name(name):
    return _normalize_text(name).replace("Advanced/Experimental Focus - ", "").strip()

def _fetch_detailed_sections_for_emphasis(program_label, emphasis_name):
    program_type, program_name = _parse_program_label(program_label)
    normalized_name = program_name.upper()
    for abbr, full_name in MAJOR_ALIASES.items():
        normalized_name = re.sub(rf"\b{abbr}\b", full_name, normalized_name)
    upper_name = normalized_name

    # Business: use emphasis-specific page for most-complete requirement sections.
    if "BUSINESS" in upper_name or "ACCOUNTANCY" in upper_name:
        url = _best_business_emphasis_url(emphasis_name)
        if url:
            try:
                resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
                if resp.status_code == 200:
                    soup = BeautifulSoup(resp.text, "html.parser")
                    page_emphasis_name = _extract_business_emphasis_name(soup) or emphasis_name
                    parsed_sections = _parse_business_requirements_from_text(soup, page_emphasis_name)
                    if parsed_sections:
                        return parsed_sections, page_emphasis_name
                    return _scrape(resp.text, "BUSINESS", url, summarize_emphasis=False), page_emphasis_name
            except requests.RequestException:
                pass

    # Generic path for Data Science and any other programs.
    dynamic = _discover_dynamic_catalog_urls(normalized_name, program_type=program_type)
    core_url = dynamic.get("CORE")
    if not core_url:
        return {}, emphasis_name
    try:
        resp = requests.get(core_url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
        if resp.status_code != 200:
            return {}, emphasis_name
        return _scrape(resp.text, normalized_name, core_url, summarize_emphasis=False), emphasis_name
    except requests.RequestException:
        return {}, emphasis_name

def _classify_emphasis_section(section_name, emphasis_name):
    business_kind = _classify_business_emphasis_section(section_name, emphasis_name)
    if business_kind == "required" or business_kind == "support":
        return "required"
    if business_kind == "additional":
        return "optional"

    section = _normalize_text(section_name)
    if ": " in section:
        section = section.split(": ", 1)[1].strip()
    section_upper = section.upper()
    emphasis_tokens = set(_name_tokens(_normalize_focus_name(emphasis_name)))
    section_tokens = set(_name_tokens(_normalize_focus_name(section)))
    overlap = len(emphasis_tokens & section_tokens)

    if "ADVANCED/EXPERIMENTAL FOCUS" in section_upper and overlap > 0:
        return "optional"
    if any(k in section_upper for k in ["FOCUS", "ELECTIVE", "OPTION", "EMPHASIS"]) and overlap > 0:
        return "optional"
    return ""

@st.cache_data(ttl=21600, show_spinner=False)
def get_emphasis_course_options(program_label, emphasis_name):
    section_map, normalized_emphasis = _fetch_detailed_sections_for_emphasis(program_label, emphasis_name)
    if not section_map:
        return []

    options = []
    seen = set()

    def _collect(section_map):
        for section, courses in section_map.items():
            section_name = _normalize_text(section)
            section_kind = _classify_emphasis_section(section_name, normalized_emphasis)
            if section_kind not in {"required", "optional"}:
                continue

            # Group OR alternatives together so each requirement appears as one choice row.
            grouped = []
            for code, title, credits in courses:
                if code.startswith("OR ") and grouped:
                    grouped[-1].append((code, title, credits))
                else:
                    grouped.append([(code, title, credits)])

            for group in grouped:
                cleaned_codes = [code.replace("OR ", "").strip() for code, _, _ in group if code.replace("OR ", "").strip()]
                if not cleaned_codes:
                    continue

                display_code = " OR ".join(cleaned_codes)
                display_title = next((title for _, title, _ in group if title), "")
                base_credits = next((credits for code, _, credits in group if not code.startswith("OR ")), group[0][2])

                dedupe_key = (section_name, display_code)
                if dedupe_key in seen:
                    continue
                seen.add(dedupe_key)
                options.append(
                    {
                        "section": section_name,
                        "code": display_code,
                        "title": display_title,
                        "credits": base_credits,
                        "section_type": section_kind,
                    }
                )
    _collect(section_map)
    return options

def _business_section_label(section_name, program_name=""):
    section = _normalize_text(section_name)
    section_upper = section.upper()

    if re.search(r"(ADMISSION|PRE-BUSINESS|LOWER LEVEL|UPPER LEVEL)", section_upper):
        return "Upper-Level Admission Requirements"
    if re.search(r"(REQUIRED CORE|BUSINESS CORE|CORE COURSES|COMMON BODY OF KNOWLEDGE)", section_upper):
        return "Business Core Requirements"
    if program_name:
        return f"{program_name}: {section}"
    return section

def _parse_business_requirements_from_text(soup, program_name):
    lines = [_normalize_text(text) for text in soup.stripped_strings]
    start_idx = next((i for i, line in enumerate(lines) if line == "Major Program Requirements"), None)
    if start_idx is None:
        return {}

    end_markers = {"Semester Plan", "Degree Audit", "Major and Career Exploration"}
    block = []
    for line in lines[start_idx + 1:]:
        if line in end_markers:
            break
        if line:
            block.append(line)

    if not block:
        return {}

    parsed_sections = {}
    current_section = None

    section_line_re = re.compile(r"^(?P<label>.+?(?:Courses|Requirements))\s+(?P<credits>\d+(?:\.\d+)?)$")
    course_re = re.compile(r"^(?P<code>[A-Z][A-Z_&\s]+ \d{4}[A-Z]?)$")
    or_course_re = re.compile(r"^or\s+(?P<code>[A-Z][A-Z_&\s]+ \d{4}[A-Z]?)$", re.IGNORECASE)

    i = 0
    while i < len(block):
        line = block[i]
        section_match = section_line_re.match(line)
        if section_match:
            current_section = _business_section_label(section_match.group("label"), program_name)
            parsed_sections.setdefault(current_section, [])
            i += 1
            continue

        if not current_section:
            i += 1
            continue

        code = None
        is_alt = False
        or_match = or_course_re.match(line)
        if or_match:
            code = _normalize_text(or_match.group("code")).upper()
            is_alt = True
        else:
            course_match = course_re.match(line)
            if course_match:
                code = _normalize_text(course_match.group("code")).upper()

        if code:
            title = ""
            if i + 1 < len(block):
                next_line = block[i + 1]
                if not section_line_re.match(next_line) and not course_re.match(next_line) and not or_course_re.match(next_line):
                    title = next_line
                    i += 1
            clean_code = f"OR {code}" if is_alt else code
            if not any(existing[0] == clean_code for existing in parsed_sections[current_section]):
                parsed_sections[current_section].append((clean_code, title, 3.0))

        i += 1

    return {name: courses for name, courses in parsed_sections.items() if courses}

def _parse_business_upper_level_from_text(soup):
    lines = [_normalize_text(text) for text in soup.stripped_strings]
    start_idx = next((i for i, line in enumerate(lines) if re.match(r"^Upper Level Admission Courses\s+\d+(?:\.\d+)?$", line, re.IGNORECASE)), None)
    if start_idx is None:
        return []

    section_line_re = re.compile(r"^.+(?:Courses|Requirements)\s+\d+(?:\.\d+)?$", re.IGNORECASE)
    course_re = re.compile(r"^([A-Z][A-Z_&\s]+ \d{4}[A-Z]?)$")
    or_course_re = re.compile(r"^or\s+([A-Z][A-Z_&\s]+ \d{4}[A-Z]?)$", re.IGNORECASE)

    courses = []
    i = start_idx + 1
    while i < len(lines):
        line = lines[i]
        if section_line_re.match(line):
            break

        code = None
        is_alt = False
        or_match = or_course_re.match(line)
        if or_match:
            code = _normalize_text(or_match.group(1)).upper()
            is_alt = True
        else:
            course_match = course_re.match(line)
            if course_match:
                code = _normalize_text(course_match.group(1)).upper()

        if code:
            title = ""
            if i + 1 < len(lines):
                next_line = lines[i + 1]
                if not section_line_re.match(next_line) and not course_re.match(next_line) and not or_course_re.match(next_line):
                    title = next_line
                    i += 1
            clean_code = f"OR {code}" if is_alt else code
            if not any(existing[0] == clean_code for existing in courses):
                courses.append((clean_code, title, 3))
        i += 1

    return courses

def _parse_business_emphasis_summary_credits(soup, emphasis_name):
    page_text = _normalize_text(soup.get_text(" ", strip=True)).upper()
    emphasis_upper = emphasis_name.upper()
    emphasis_pattern = _build_emphasis_phrase_pattern(emphasis_name)

    patterns = [
        rf"REQUIRED\s+{emphasis_pattern}\s+COURSES\s+(\d+(?:\.\d+)?)" if emphasis_pattern else "",
        rf"REQUIRED\s+{emphasis_pattern}\s+EMPHASIS\s+COURSES\s+(\d+(?:\.\d+)?)" if emphasis_pattern else "",
        rf"ADDITIONAL\s+{emphasis_pattern}\s+COURSES\s+(\d+(?:\.\d+)?)" if emphasis_pattern else "",
        rf"ADDITIONAL\s+{emphasis_pattern}\s+EMPHASIS\s+COURSES\s+(\d+(?:\.\d+)?)" if emphasis_pattern else "",
        r"EMPHASIS\s+SUPPORT\s+COURSES\s+(\d+(?:\.\d+)?)",
    ]

    total = 0.0
    matched = False
    for pattern in patterns:
        if not pattern:
            continue
        match = re.search(pattern, page_text, re.IGNORECASE)
        if match:
            total += float(match.group(1))
            matched = True

    if matched:
        return total

    lines = [_normalize_text(text) for text in soup.stripped_strings]
    start_idx = next((i for i, line in enumerate(lines) if line == "Major Program Requirements"), None)
    if start_idx is None:
        return 0

    block = []
    for line in lines[start_idx + 1:]:
        if line in {"Semester Plan", "Degree Audit", "Major and Career Exploration"}:
            break
        if line:
            block.append(line)

    section_credit_re = re.compile(r"^(?P<label>.+?)\s+(?P<credits>\d+(?:\.\d+)?)$")
    fallback_total = 0.0
    for line in block:
        match = section_credit_re.match(line)
        if not match:
            continue

        label_upper = match.group("label").upper()
        credits = float(match.group("credits"))

        if label_upper in {f"REQUIRED {emphasis_upper} COURSES", f"REQUIRED {emphasis_upper} EMPHASIS COURSES"}:
            fallback_total += credits
        elif label_upper in {f"ADDITIONAL {emphasis_upper} COURSES", f"ADDITIONAL {emphasis_upper} EMPHASIS COURSES"}:
            fallback_total += credits
        elif label_upper == "EMPHASIS SUPPORT COURSES":
            fallback_total += credits

    return fallback_total

def _clean_program_label(label):
    cleaned = _normalize_text(label)
    cleaned = re.sub(r"^(Undergraduate|Graduate)\s+", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"^(Major|Minor|Certificate|Emphasis)\s+in\s+", "", cleaned, flags=re.IGNORECASE)
    return cleaned

def _extract_link_pairs(soup, base_url):
    pairs = []
    for a in soup.find_all("a", href=True):
        text = _normalize_text(a.get_text(" ", strip=True))
        if not text:
            continue
        href = a["href"]
        if not href.startswith("http"):
            href = requests.compat.urljoin(base_url, href)
        if "catalog.missouri.edu" not in href or href.endswith(".pdf"):
            continue
        pairs.append((text, href))
    return pairs

@lru_cache(maxsize=1)
def _build_catalog_program_index():
    index_entries = []
    headers = {"User-Agent": "Mozilla/5.0"}

    # 1) Degrees + emphasis index
    try:
        resp = requests.get(DEGREE_INDEX_URL, headers=headers, timeout=10)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        for row in soup.find_all("tr"):
            cells = row.find_all(["td", "th"])
            if len(cells) < 2:
                continue
            primary_name = _normalize_text(cells[0].get_text(" ", strip=True))
            for _, href in _extract_link_pairs(row, DEGREE_INDEX_URL):
                if "/courseofferings/" in href:
                    continue
                index_entries.append({"name": primary_name, "type": "major", "url": href})
    except requests.RequestException:
        pass

    # 2) Minors + certificates index (crawl one level to college pages)
    try:
        resp = requests.get(ADDITIONAL_INDEX_URL, headers=headers, timeout=10)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        college_pages = []
        for text, href in _extract_link_pairs(soup, ADDITIONAL_INDEX_URL):
            if "additionalcertificatesminors" in href:
                college_pages.append(href)
            elif re.match(r"^(Minor|Certificate)\s+in\s+", text, re.IGNORECASE):
                ptype = "minor" if text.upper().startswith("MINOR") else "certificate"
                index_entries.append({"name": _clean_program_label(text), "type": ptype, "url": href})

        for page_url in sorted(set(college_pages)):
            try:
                page_resp = requests.get(page_url, headers=headers, timeout=10)
                page_resp.raise_for_status()
                page_soup = BeautifulSoup(page_resp.text, "html.parser")
                for text, href in _extract_link_pairs(page_soup, page_url):
                    if re.match(r"^Minor in\s+.+", text, re.IGNORECASE):
                        index_entries.append({"name": _clean_program_label(text), "type": "minor", "url": href})
                    elif re.match(r"^(Undergraduate|Graduate)?\s*Certificate in\s+.+", text, re.IGNORECASE):
                        index_entries.append({"name": _clean_program_label(text), "type": "certificate", "url": href})
                    elif re.match(r"^Emphasis in\s+.+", text, re.IGNORECASE):
                        index_entries.append({"name": _clean_program_label(text), "type": "emphasis", "url": href})
            except requests.RequestException:
                continue
    except requests.RequestException:
        pass

    deduped = {}
    for entry in index_entries:
        if not entry["name"] or not entry["url"]:
            continue
        key = (entry["type"], _normalize_lookup_key(entry["name"]), entry["url"])
        deduped[key] = entry
    return list(deduped.values())

def _discover_dynamic_catalog_urls(program_str, program_type="major"):
    try:
        entries = _build_catalog_program_index()
    except Exception:
        return {}

    program_key = _normalize_lookup_key(program_str)
    query_tokens = set(_name_tokens(program_str))
    target_type = (program_type or "").lower()

    candidates = []
    for entry in entries:
        entry_key = _normalize_lookup_key(entry["name"])
        entry_tokens = set(_name_tokens(entry["name"]))
        overlap = len(query_tokens & entry_tokens)
        if overlap == 0 and program_key not in entry_key and entry_key not in program_key:
            continue

        exact_bonus = 5 if entry_key == program_key else 0
        type_bonus = 2 if target_type and entry["type"] == target_type else 0
        score = overlap + exact_bonus + type_bonus
        candidates.append((score, entry))

    if not candidates:
        return {}

    best_match = max(candidates, key=lambda item: item[0])[1]
    return {"CORE": best_match["url"]}

def get_requirements(program_str, emphases=None, program_type="major"):
    if emphases is None:
        emphases = []
    major_upper = program_str.upper()

    for abbr, full_name in MAJOR_ALIASES.items():
        major_upper = re.sub(rf"\b{abbr}\b", full_name, major_upper)

    aggregated_reqs = {}

    for keyword, url_dict in CATALOG_URLS.items():
        if keyword in major_upper:
            # Check if any detected emphasis matches our known URLs
            known_emp_keys = [k for k in url_dict.keys() if k in emphases and k != "CORE"]
            
            if known_emp_keys:
                # Student has a specific emphasis! Scrape strictly that URL + Core
                keys_to_scrape = known_emp_keys + (["CORE"] if "CORE" in url_dict else [])
                for key in set(keys_to_scrape):
                    try:
                        url = url_dict[key]
                        resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
                        if resp.status_code == 200:
                            should_summarize = (keyword == "BUSINESS" and key == "CORE")
                            scraped = _scrape(resp.text, major_upper, url, summarize_emphasis=should_summarize)
                            _merge_reqs(aggregated_reqs, scraped)
                    except: continue
            else:
                # No emphasis detected! Scrape ALL URLs for this major, but summarize the emphasis sections.
                for key, url in url_dict.items():
                    try:
                        resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
                        if resp.status_code == 200:
                            scraped = _scrape(resp.text, major_upper, url, summarize_emphasis=True)
                            _merge_reqs(aggregated_reqs, scraped)
                    except: continue
            
            if aggregated_reqs:
                return aggregated_reqs

    dynamic_urls = _discover_dynamic_catalog_urls(program_str, program_type=program_type)
    if dynamic_urls.get("CORE"):
        try:
            resp = requests.get(dynamic_urls["CORE"], headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
            if resp.status_code == 200:
                return _scrape(resp.text, major_upper, dynamic_urls["CORE"], summarize_emphasis=False)
        except:
            pass
    return {}

def _scrape(html, major_string, url="", summarize_emphasis=False):
    soup = BeautifulSoup(html, "html.parser")
    sections = {}
    current = "Requirements"
    skip_section = False

    for tag in soup.find_all(["h2", "h3", "h4", "tr"]):
        # 1. Parse high-level headers
        if tag.name in ("h2", "h3", "h4"):
            t = tag.get_text(strip=True)
            if t and len(t) < 80:
                t_upper = t.upper()
                
                if any(x in t_upper for x in ["SEMESTER PLAN", "PLAN OF STUDY", "ONLINE", "SUMMARY", "GRADUATE", "MASTER", "MACC", "HONORS", "ACCELERATED"]):
                    skip_section = True
                    continue
                
                is_ba = "B.A." in t_upper or "BACHELOR OF ARTS" in t_upper
                is_bs = "B.S." in t_upper or "BACHELOR OF SCIENCE" in t_upper
                
                if is_ba and "BS" in major_string: skip_section = True
                elif is_bs and "BA" in major_string: skip_section = True
                else: skip_section = False 
                    
                current = t
            continue
            
        if skip_section: continue

        # 2. Parse CourseLeaf Tables dynamically
        if tag.name == "tr":
            comment = tag.find("span", class_="courselistcomment")
            if comment:
                c_text = comment.get_text(strip=True)
                if len(c_text) > 3:
                    c_up = c_text.upper()
                    if any(x in c_up for x in ["GRADUATE", "MASTER", "MACC"]):
                        skip_section = True
                        continue
                    else:
                        skip_section = False
                        current = c_text
                continue
            
            is_alt = 'orclass' in tag.get('class', [])
            links = tag.find_all("a")
            for a in links:
                code_raw = a.get_text(strip=True).upper().replace("\xa0", " ").replace("_", " ")
                match = re.search(r"([A-Z\s]+\d{4})", code_raw)
                if not match: continue
                    
                clean_code = match.group(1).strip()
                
                # CRITICAL: Ignore graduate level courses (7000+)
                num_match = re.search(r"\d{4}", clean_code)
                if num_match and int(num_match.group()) >= 7000:
                    continue 
                
                td = a.find_parent("td")
                if td and td.get_text(strip=True).lower().startswith("or"):
                    is_alt = True
                    
                if is_alt: clean_code = "OR " + clean_code
                    
                tds = tag.find_all("td")
                title = ""
                for i, td_elem in enumerate(tds):
                    if a in td_elem.find_all("a") and i + 1 < len(tds):
                        title = tds[i + 1].get_text(strip=True)
                        break
                        
                sections.setdefault(current, [])
                if not any(c[0] == clean_code for c in sections[current]):
                    sections[current].append((clean_code, title, 3))

    # 3. DYNAMIC GROUPING & SUMMARIZATION
    processed = {}
    major_upper = major_string.upper()
    
    if "BUSINESS" in major_upper:
        emphasis_name = _extract_business_emphasis_name(soup, sections.keys())

        if not any(any(k in sec.upper() for k in ["ADMISSION", "UPPER-LEVEL", "UPPER LEVEL"]) for sec in sections):
            upper_level_courses = _parse_business_upper_level_from_text(soup)
            if upper_level_courses:
                sections["Upper-Level Admission Requirements"] = upper_level_courses
        
        emphasis_credits = 0
        
        for sec, courses in sections.items():
            s_up = sec.upper()
            if any(x in s_up for x in ["ONLINE", "SUMMARY", "GRADUATE", "MASTER", "MINOR"]): continue
                
            if any(k in s_up for k in ["ADMISSION", "UPPER-LEVEL", "UPPER LEVEL"]):
                processed.setdefault("Upper-Level Admission Requirements", []).extend(c for c in courses if c not in processed.get("Upper-Level Admission Requirements", []))
                            
            elif any(k in s_up for k in ["REQUIRED CORE", "BUSINESS CORE", "REQUIRED BUSINESS", "CORE REQUIREMENTS"]):
                processed.setdefault("Business Core Requirements", []).extend(c for c in courses if c not in processed.get("Business Core Requirements", []))
                            
            elif _is_business_emphasis_section(sec, emphasis_name):
                if summarize_emphasis:
                    emphasis_credits += _sum_required_credits(courses)
                else:
                    new_sec_name = f"{emphasis_name} Emphasis: {sec}" if emphasis_name else sec
                    processed.setdefault(new_sec_name, []).extend(c for c in courses if c not in processed.get(new_sec_name, []))
            elif not summarize_emphasis:
                processed.setdefault(sec, []).extend(c for c in courses if c not in processed.get(sec, []))
                
        # Generate summary row if flag is true
        if summarize_emphasis and emphasis_name:
            parsed_summary_credits = _parse_business_emphasis_summary_credits(soup, emphasis_name)
            if parsed_summary_credits > 0:
                emphasis_credits = parsed_summary_credits

        if summarize_emphasis and emphasis_name and emphasis_credits > 0:
            processed[f"{emphasis_name} Emphasis Options"] = [("EMPHASIS", f"Complete {emphasis_credits} credits of {emphasis_name} requirements", emphasis_credits)]
                
    elif "DATA SCIENCE" in major_upper:
        for sec, courses in sections.items():
            s_up = sec.upper()
            if "ONLINE" in s_up: continue
            
            if "CORE" in s_up:
                processed.setdefault("Core Courses", []).extend(c for c in courses if c not in processed.get("Core Courses", []))
            elif "INTERMEDIATE" in s_up:
                processed.setdefault("Intermediate Courses", []).extend(c for c in courses if c not in processed.get("Intermediate Courses", []))
            elif any(x in s_up for x in ["ADVANCED", "EXPERIENTIAL", "EXPERIMENTAL", "FOCUS", "ELECTIVE", "EMPHASIS"]):
                group_name = "Advanced/Experimental Focus"
                if "STAT" in s_up: group_name = "Advanced/Experimental Focus - Statistics"
                elif "MATH" in s_up: group_name = "Advanced/Experimental Focus - Mathematics"
                elif "COMP" in s_up or "CS" in s_up: group_name = "Advanced/Experimental Focus - Computer Science"
                
                if summarize_emphasis:
                    total_credits = _sum_required_credits(courses)
                    processed[group_name] = [("EMPHASIS", f"Select {total_credits} credits for {group_name}", total_credits)]
                else:
                    processed.setdefault(group_name, []).extend(c for c in courses if c not in processed.get(group_name, []))
            elif not summarize_emphasis:
                processed.setdefault(sec, []).extend(c for c in courses if c not in processed.get(sec, []))
    else:
        processed = sections

    return processed

# ─────────────────────────────────────────────────────────────────
# GAP ANALYSIS
# ─────────────────────────────────────────────────────────────────
def norm(code):
    code = code.upper().replace("_", " ").strip()
    return re.sub(r"(\d+)[HW]\b", r"\1", code)

def gap_analysis(courses, requirements):
    done = {norm(c["code"]): c for c in courses if c["status"] == "Complete"}
    prog = {norm(c["code"]): c for c in courses if c["status"] == "In Progress"}
    results = []
    
    for section, items in requirements.items():
        sec_upper = section.upper()
        is_pick_list = any(k in sec_upper for k in ["FOCUS", "EXPERIENTIAL", "ELECTIVE", "CHOOSE", "SELECT", "EMPHASIS", "OPTIONS"])
        
        groups = []
        for code, title, _ in items:
            if code.startswith("OR ") and groups:
                groups[-1].append((code, title))
            else:
                groups.append([(code, title)])
                
        for group in groups:
            completed_course = None
            inprogress_course = None
            
            for code, title in group:
                search_code = norm(code).replace("OR ", "")
                
                if search_code in done:
                    completed_course = (code, title, "✅ Complete", done[search_code]["grade"])
                    break  
                elif search_code in prog and not inprogress_course:
                    inprogress_course = (code, title, "⏳ In Progress", "IP")
            
            if completed_course:
                c_code, c_title, c_status, c_grade = completed_course
                results.append({"section": section, "code": c_code.replace("OR ", ""), "title": c_title, "status": c_status, "grade": c_grade})
            elif inprogress_course:
                i_code, i_title, i_status, i_grade = inprogress_course
                results.append({"section": section, "code": i_code.replace("OR ", ""), "title": i_title, "status": i_status, "grade": i_grade})
            else:
                first_code, first_title = group[0]
                status_text = "⚪ Option" if is_pick_list else "❌ Outstanding"
                
                # Catch the dynamically generated summary rows
                if first_code.replace("OR ", "") == "EMPHASIS":
                    status_text = "ℹ️ Summary"
                    
                results.append({"section": section, "code": first_code.replace("OR ", ""), "title": first_title, "status": status_text, "grade": "—"})

    return results

def _split_audit_results(results):
    detailed_sections = {}
    emphasis_summaries = []

    for row in results:
        if row["code"] == "EMPHASIS" and "Summary" in row["status"]:
            emphasis_summaries.append(row)
        else:
            detailed_sections.setdefault(row["section"], []).append(row)

    return detailed_sections, emphasis_summaries

def _style_status_table(df):
    def color_status(row):
        status = row["status"]
        if "✅" in status:
            return ["background-color: #d4edda; color: #155724; font-weight: bold"] * len(row)
        if "⏳" in status:
            return ["background-color: #fff3cd; color: #856404; font-weight: bold"] * len(row)
        if "❌" in status:
            return ["background-color: #f8d7da; color: #721c24; font-weight: bold"] * len(row)
        if "ℹ️" in status:
            return ["background-color: #e2e3e5; color: #383d41; font-weight: bold"] * len(row)
        return [""] * len(row)

    return df.style.apply(color_status, axis=1).hide(axis="index")

def render_degree_audit(major, results, emphases):
    detailed_sections, emphasis_summaries = _split_audit_results(results)
    active_emphasis = ", ".join(emphases) if emphases else None

    heading = f"#### **Degree Audit for: <span style='color:{MIZZOU_GOLD};'>{major}**</span>"
    if active_emphasis:
        heading += f"  \n**Detected emphasis:** {active_emphasis}"
    st.markdown(heading, unsafe_allow_html=True)

    if emphasis_summaries and not active_emphasis:
        st.caption("No emphasis area was found on the transcript, so the audit shows the shared requirements plus a credit-only summary for each emphasis option.")
        summary_df = pd.DataFrame(
            [
                {
                    "Emphasis Area": row["section"].replace(" Emphasis Options", "").replace("Advanced/Experimental Focus - ", ""),
                    "Requirement Summary": row["title"],
                }
                for row in emphasis_summaries
            ]
        )
        st.dataframe(summary_df, use_container_width=True, hide_index=True)

        for idx, row in enumerate(emphasis_summaries):
            emphasis_area = row["section"].replace(" Emphasis Options", "").replace("Advanced/Experimental Focus - ", "").strip()
            with st.expander(f"Course Options: {emphasis_area}", expanded=False):
                options = get_emphasis_course_options(major, emphasis_area)
                if not options:
                    st.caption("No option list could be loaded for this emphasis from the live catalog.")
                    continue

                option_rows = pd.DataFrame(
                    [
                        {
                            "code": opt["code"],
                            "title": opt["title"] if opt["title"] else "",
                            "status": "ℹ️ Required" if opt.get("section_type") == "required" else "⚪ Optional",
                            "grade": "—",
                        }
                        for opt in options
                    ]
                )
                st.dataframe(_style_status_table(option_rows), use_container_width=True)

    for section, rows in detailed_sections.items():
        section_df = pd.DataFrame(rows)[["code", "title", "status", "grade"]]
        open_by_default = any("❌" in row["status"] or "⏳" in row["status"] for row in rows)
        with st.expander(section, expanded=open_by_default):
            st.dataframe(_style_status_table(section_df), use_container_width=True)

# ─────────────────────────────────────────────────────────────────
# AI TOOLS (LIVE SCRAPING & EXPORT)
# ─────────────────────────────────────────────────────────────────
def _normalize_course_code(course_code: str) -> str:
    code = course_code.upper().replace("_", " ").strip()
    code = re.sub(r"\s+", " ", code)
    return code

def _extract_prerequisite_text(course_text: str) -> str:
    flat = _normalize_text(course_text)
    match = re.search(
        r"Prerequisite[s]?:\s*(.*?)(?:\s*(?:Corequisite[s]?:|Recommended:|Credit(s)?\s*:|$))",
        flat,
        flags=re.IGNORECASE,
    )
    return _normalize_text(match.group(1)) if match else ""

def get_course_details(course_code: str) -> dict:
    clean_code = _normalize_course_code(course_code)
    match = re.match(r"^([A-Z][A-Z&\s]+)\s+(\d{4}[A-Z]?)$", clean_code)
    if not match:
        return {"error": f"Invalid course code format: {clean_code}"}
        
    subject = match.group(1).strip()
    number = match.group(2).strip()
    target_code = _normalize_course_code(f"{subject} {number}")
    subject_url = subject.replace(" ", "_").lower()
    url = f"https://catalog.missouri.edu/courseofferings/{subject_url}/"
    
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        resp = requests.get(url, headers=headers, timeout=8)
        if resp.status_code != 200:
            return {"error": f"Could not find catalog page for {subject}."}
            
        soup = BeautifulSoup(resp.text, "html.parser")
        blocks = soup.find_all("div", class_="courseblock")

        for block in blocks:
            title_node = block.find("p", class_="courseblocktitle")
            title_text = _normalize_text(title_node.get_text(" ", strip=True) if title_node else block.get_text(" ", strip=True))
            code_match = re.search(r"([A-Z][A-Z&\s_]+)\s+(\d{4}[A-Z]?)", title_text.upper())
            if not code_match:
                continue

            block_code = _normalize_course_code(f"{code_match.group(1).replace('_', ' ')} {code_match.group(2)}")
            if block_code == target_code:
                full_text = block.get_text(separator=" ", strip=True)
                prereq_text = _extract_prerequisite_text(full_text)
                return {
                    "course": target_code,
                    "catalog_data": _normalize_text(full_text),
                    "prerequisites": prereq_text or "No explicit prerequisite listed.",
                }
                
        return {"error": f"Course {target_code} not found on the live {subject} catalog page."}
    except Exception as e:
        return {"error": f"Failed to fetch course details: {str(e)}"}

def export_schedule(course_codes: list[str]) -> str:
    filename = "mizzou_semester_schedule.csv"
    try:
        with open(filename, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(["Course Code", "Status"])
            for code in course_codes:
                writer.writerow([code.upper(), "Planned"])
                
        return f"SUCCESS: Schedule saved locally to {filename}!"
    except Exception as e:
        return f"ERROR: Could not save schedule. {e}"

def _build_compact_gap_context(gap_analysis_results, max_items_per_major=30):
    compact = {}
    for major, rows in (gap_analysis_results or {}).items():
        complete = [r for r in rows if "✅" in r.get("status", "")]
        in_progress = [r for r in rows if "⏳" in r.get("status", "")]
        outstanding = [r for r in rows if "❌" in r.get("status", "")]
        summaries = [r for r in rows if "ℹ️" in r.get("status", "")]

        # Keep context focused on actionable items to reduce token usage.
        top_outstanding = [
            {
                "section": r.get("section", ""),
                "code": r.get("code", ""),
                "title": r.get("title", ""),
                "status": r.get("status", ""),
            }
            for r in outstanding[:max_items_per_major]
        ]
        top_in_progress = [
            {
                "section": r.get("section", ""),
                "code": r.get("code", ""),
                "title": r.get("title", ""),
                "status": r.get("status", ""),
            }
            for r in in_progress[:10]
        ]
        emphasis_summaries = [
            {
                "section": r.get("section", ""),
                "title": r.get("title", ""),
            }
            for r in summaries[:10]
        ]
        in_progress_codes = sorted({r.get("code", "") for r in in_progress if r.get("code", "") and r.get("code", "") != "EMPHASIS"})
        outstanding_codes = sorted({r.get("code", "") for r in outstanding if r.get("code", "") and r.get("code", "") != "EMPHASIS"})

        compact[major] = {
            "counts": {
                "complete": len(complete),
                "in_progress": len(in_progress),
                "outstanding": len(outstanding),
                "summary_rows": len(summaries),
                "total_rows": len(rows),
            },
            "top_outstanding": top_outstanding,
            "top_in_progress": top_in_progress,
            "in_progress_codes_all": in_progress_codes,
            "outstanding_codes_all": outstanding_codes,
            "emphasis_summaries": emphasis_summaries,
        }
    return compact

# ─────────────────────────────────────────────────────────────────
# STREAMLIT WEB INTERFACE & AI CHAT
# ─────────────────────────────────────────────────────────────────
def init_chat_session(transcript_data, eligible_courses):
    advisor_persona = """
    You are an expert academic advisor for the University of Missouri (Mizzou). 
    Your goal is to help students navigate their degree path based strictly on the gap analysis provided.
    
    CRITICAL RULES:
    1. Prerequisite Checking: Before recommending a course, verify prerequisites using live course catalog lookup.
    3. Use Mizzou-themed language when appropriate (e.g., refer to 'myZou' or use 'Tiger' metaphors).
    4. Exporting: When the student agrees on a final list of classes for next semester, export a CSV schedule for them.
    5. Never mention internal function or tool names in your responses.
    6. Never recommend a course for a future semester if it is already marked In Progress in the student's record.
    7. Do not recommend a course and one of its prerequisites in the same semester plan.
    SCHEDULING RULES: 1. A semester schedule MUST contain between 12 and 18 credit hours. NEVER exceed 18 hours. 2. You may only recommend a course if the student has completed the required prerequisites. 3. Output the recommended schedule as a simple bulleted list with the course code and credit hours.
    """

    program_labels = [f"{p['type'].title()}: {p['name']}" for p in transcript_data.get("programs", [])]
    student_context = f"""
    Here is the student's academic profile:
    Name: {transcript_data['name']}
    Majors: {', '.join(transcript_data['majors'])}
    Programs: {', '.join(program_labels) if program_labels else 'None'}
    Emphases: {', '.join(transcript_data['emphases']) if transcript_data['emphases'] else 'None'}
    GPA: {transcript_data['gpa']}
    
    Eligible Courses Next Semester:
    {json.dumps(eligible_courses, indent=2)}

    Interpret any course listed under `in_progress_codes_all` as unavailable for future-semester recommendations.
    Also avoid placing a course in the same semester as any of its prerequisites.
    
    Introduce yourself to the student (using a Black & Gold Tiger flavor) and ask what term they are scheduling for.
    """
    client = Groq(api_key=st.secrets["GROQ_API_KEY"])
    chat = TigerChat(client, advisor_persona)
    
    try:
        initial_response = chat.send_message(student_context)
        return client, chat, initial_response.text
    except Exception as e:
        error_msg = f"⚠️ **API Error:** Details: {e}"
        return client, chat, error_msg

def get_eligible_courses(gap_analysis, transcript_courses):
    # TODO: Apply a Directed Acyclic Graph (DAG) prerequisite filter here to remove missing courses whose prerequisites are not yet met.
    return gap_analysis

def _upload_signature(uploaded_file):
    if uploaded_file is None:
        return None
    return f"{uploaded_file.name}:{getattr(uploaded_file, 'size', 0)}"

def main():
    chat_tab, data_tab = st.tabs(["Advisor Chat", "Degree Audit"])

    with chat_tab:
        header_left, header_right = st.columns([0.78, 0.22])
        with header_left:
            st.markdown("#### Advisor Chat")
        with header_right:
            uploaded_file = st.file_uploader(
                "Upload Transcript PDF",
                type="pdf",
                key="chat_upload",
                label_visibility="collapsed",
                help="Upload your unofficial Mizzou transcript.",
            )

    current_sig = _upload_signature(uploaded_file)
    previous_sig = st.session_state.get("active_upload_sig")
    if current_sig != previous_sig:
        for key in ["analyzed", "transcript", "gap_analysis", "chat", "chat_client", "messages", "pending_user_input"]:
            st.session_state.pop(key, None)
        st.session_state.active_upload_sig = current_sig

    has_upload = uploaded_file is not None
    is_initialized = all(
        key in st.session_state
        for key in ["analyzed", "transcript", "gap_analysis", "chat", "chat_client", "messages"]
    )
    if has_upload and not is_initialized:
        with st.spinner("Parsing transcript and analyzing catalog requirements..."):
            t = parse_transcript(uploaded_file)
            st.session_state.transcript = t

            all_results = {}
            programs = t.get("programs", [])
            if not programs and t["majors"]:
                programs = [{"type": "major", "name": m} for m in t["majors"]]

            for program in programs:
                reqs = get_requirements(
                    program["name"],
                    emphases=t["emphases"],
                    program_type=program.get("type", "major"),
                )
                if reqs:
                    label = f"{program.get('type', 'program').title()}: {program['name']}"
                    all_results[label] = gap_analysis(t["courses"], reqs)

            gap = all_results
            eligible_courses = get_eligible_courses(gap, t["courses"])
            chat_client, chat_session, first_msg = init_chat_session(t, eligible_courses)

            st.session_state.gap_analysis = all_results
            st.session_state.analyzed = True
            st.session_state.chat_client = chat_client
            st.session_state.chat = chat_session
            st.session_state.messages = [{"role": "assistant", "content": first_msg}]

    with data_tab:
        if not has_upload or "transcript" not in st.session_state:
            st.info("Upload a transcript from the chat panel to generate a degree audit.")
        else:
            t = st.session_state.transcript
            st.markdown(f"### **Mizzou Academic Profile: {t['name']}**")

            col1, col2, col3 = st.columns(3)
            col1.metric("Current CUM GPA", f"{t['gpa']}", help="From myZou")
            col2.metric("Credits Earned", f"{int(t['hours'])}", "Fall 2024 CUM")
            col3.metric("Enrollment Status", "Active", help="Spring 2025 semester")
            st.divider()

            if st.session_state.gap_analysis:
                for major, results in st.session_state.gap_analysis.items():
                    render_degree_audit(major, results, t["emphases"])
            else:
                st.warning("No academic program detected on transcript. AI will generalize.")

    with chat_tab:
        if not has_upload or "messages" not in st.session_state:
            st.info("Upload your transcript using the top-right control to start the advisor chat.")
            st.chat_input("Ask a question about scheduling...", disabled=True)
            return

        t = st.session_state.transcript
        pending_user_input = st.session_state.pop("pending_user_input", None)
        if pending_user_input:
            with st.spinner("Thinking..."):
                try:
                    time.sleep(1)
                    response = st.session_state.chat.send_message(pending_user_input)
                    st.session_state.messages.append({"role": "assistant", "content": response.text})
                except Exception as e:
                    if "429" in str(e):
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": "Rate limit reached. Please wait a moment and try again.",
                        })
                    else:
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": f"Error communicating with the AI service: {e}",
                        })

        with st.expander("View Transcript Summary (Optional)"):
            st.write(f"**Parsed Name:** {t['name']}")
            st.write(f"**Parsed Majors:** {', '.join(t['majors']) if t['majors'] else 'None'}")
            st.write(f"**Parsed Minors:** {', '.join(t['minors']) if t.get('minors') else 'None'}")
            st.write(f"**Parsed Certificates:** {', '.join(t['certificates']) if t.get('certificates') else 'None'}")
            st.write(f"**Parsed Emphases:** {', '.join(t['emphases']) if t['emphases'] else 'None Detected (Showing Summaries)'}")

        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        if user_input := st.chat_input("Ask a question about your prerequisites or balance..."):
            st.session_state.messages.append({"role": "user", "content": user_input})
            st.session_state.pending_user_input = user_input
            st.rerun()

if __name__ == "__main__":
    main()
