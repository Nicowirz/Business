import streamlit as st
import pdfplumber
import re
import requests
from bs4 import BeautifulSoup
import json
import csv

# NEW SDK IMPORTS
from google import genai
from google.genai import types

# ─────────────────────────────────────────────────────────────────
# AI ADVISOR CONFIGURATION
# ─────────────────────────────────────────────────────────────────
# Replace this with your actual Gemini API Key
API_KEY = "AIzaSyAIvRxDHOD5TZ81EuKIWHxD7m6ZKbrUNbk"

# ─────────────────────────────────────────────────────────────────
# CATALOG URL MAP
# ─────────────────────────────────────────────────────────────────
CATALOG_URLS = {
    "BUSINESS":         "https://catalog.missouri.edu/collegeofbusiness/businessadministration/bsba-business-administration/",
    "ACCOUNTANCY":      "https://catalog.missouri.edu/collegeofbusiness/accountancy/bsacc-accountancy/",
    "DATA SCIENCE":     "https://catalog.missouri.edu/collegeofengineering/datascience/bs-data-science/",
    "COMPUTER SCIENCE": "https://catalog.missouri.edu/collegeofengineering/computerscience/bs-computer-science/",
    "MECHANICAL ENG":   "https://catalog.missouri.edu/collegeofengineering/mechanicalengineering/bs-mechanical-engineering/",
    "FINANCE":          "https://catalog.missouri.edu/collegeofbusiness/finance/bsba-finance/",
    "ECONOMICS":        "https://catalog.missouri.edu/collegeofartsandscience/economics/bs-economics/",
}

# ─────────────────────────────────────────────────────────────────
# BUILT-IN REQUIREMENTS (2025-26 catalog fallback)
# ─────────────────────────────────────────────────────────────────
BUILTIN_REQS = {
    "DATA SCIENCE": {
        "Core Courses (all required — 30 cr)": [
            ("DATA SCI 1030", "Foundations of Data Science", 3),
            ("STAT 2800",     "Intuition, Simulation, and Data", 3),
            ("CMP SC 1300",   "Computing with Data in Python", 3),
            ("CMP SC 2300",   "Intro to Computational Data Visualization", 3),
            ("CMP SC 3380",   "Database Applications and Info Systems", 3),
            ("STAT 4510",     "Applied Statistical Models I", 3),
            ("STAT 4520",     "Applied Statistical Models II", 3),
            ("MATH 1500",     "Analytic Geometry & Calculus I (or MATH 1400)", 5),
            ("MATH 2320",     "Discrete Mathematical Structures", 3),
            ("MATH 4140",     "Matrix Theory", 3),
        ],
        "Intermediate Courses (choose 4, 12 cr)": [
            ("CMP SC 4350", "Big Data Analytics", 3),
            ("CMP SC 4720", "Intro to Machine Learning", 3),
            ("STAT 4560",   "Applied Multivariate Data Analysis", 3),
            ("STAT 4640",   "Intro to Bayesian Data Analysis", 3),
            ("MATH 1700",   "Calculus II (or MATH 2100)", 5),
            ("MATH 4100",   "Differential Equations", 3),
        ],
        "Advanced Focus — CS (choose 4, 12 cr)": [
            ("CMP SC 4540", "Neural Models and Machine Learning", 3),
            ("CMP SC 4740", "Interdisciplinary Intro to NLP", 3),
            ("CMP SC 4750", "Artificial Intelligence I", 3),
            ("CMP SC 4770", "Intro to Computational Intelligence", 3),
        ],
        "Experiential Courses (choose 6 cr)": [
            ("CMP SC 4990",  "Undergraduate Research in CS", 3),
            ("INTDSC 4971",  "Capstone Internship", 3),
            ("STAT 4085",    "Problems in Statistics for Undergraduates", 3),
            ("MATH 4960",    "Special Readings in Mathematics", 3),
        ],
        "Supporting / Gen Ed": [
            ("ENGLSH 1000", "Writing and Rhetoric", 3),
            ("MATH 1100",   "College Algebra (or equivalent)", 3),
        ],
    },
    "BUSINESS": {
        "Upper-Level Admission Courses": [
            ("ACCTCY 2036", "Accounting I", 3),
            ("ACCTCY 2037", "Accounting II", 3),
            ("ACCTCY 2258", "Computer-Based Data Systems (or CMP SC 1050)", 3),
            ("BUS AD 1500", "Foundations of Business & Prof Dev Principles", 3),
            ("ECONOM 1014", "Principles of Microeconomics", 3),
            ("ECONOM 1015", "Principles of Macroeconomics", 3),
            ("ENGLSH 1000", "Writing and Rhetoric", 3),
            ("MATH 1100",   "College Algebra", 3),
            ("MATH 1400",   "Calculus for Social/Life Sciences I (or MATH 1300)", 3),
            ("STAT 2500",   "Intro to Probability and Statistics I", 3),
        ],
        "Required Business Core Courses": [
            ("BUS AD 3500",  "Advanced Professional Development Principles", 3),
            ("BUS AD 4500",  "Professional Development Program – Internship", 3),
            ("FINANC 3000",  "Corporate Finance", 3),
            ("MANGMT 3000",  "Principles of Management", 3),
            ("MANGMT 3300",  "Intro to Business Processes and Technologies", 3),
            ("MANGMT 3540",  "Introduction to Business Law", 3),
            ("MRKTNG 3000",  "Principles of Marketing", 3),
        ],
        "Capstone": [
            ("MANGMT 4970", "Strategic Management", 3),
        ],
    },
}

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
    return {
        "name":    _parse_name(lines),
        "majors":  _parse_majors(lines),
        "gpa":     _parse_gpa_hours(lines)[0],
        "hours":   _parse_gpa_hours(lines)[1],
        "courses": _parse_courses(lines),
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
# CATALOG LOOKUP
# ─────────────────────────────────────────────────────────────────

MAJOR_ALIASES = {
    "BUS OR ACCTCY": "BUSINESS",  
    "BUS": "BUSINESS",
    "ACCTCY": "ACCOUNTANCY",
    "COMP SCI": "COMPUTER SCIENCE",
    "MECH ENG": "MECHANICAL ENG"
}

def get_requirements(major_str):
    major_upper = major_str.upper()

    for abbr, full_name in MAJOR_ALIASES.items():
        major_upper = re.sub(rf"\b{abbr}\b", full_name, major_upper)

    for keyword, url in CATALOG_URLS.items():
        if keyword in major_upper:
            try:
                headers = {
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
                }
                resp = requests.get(url, headers=headers, timeout=10)
                
                if resp.status_code == 200:
                    scraped = _scrape(resp.text, major_upper)
                    if scraped:
                        return scraped
            except requests.RequestException:
                break  

    for keyword, reqs in BUILTIN_REQS.items():
        if keyword in major_upper:
            return reqs

    return {}

def _scrape(html, major_string):
    soup = BeautifulSoup(html, "html.parser")
    sections = {}
    current = "Requirements"
    skip_section = False

    for tag in soup.find_all(["h3", "h4", "tr"]):
        if tag.name in ("h3", "h4"):
            t = tag.get_text(strip=True)
            if t and len(t) < 80:
                t_upper = t.upper()
                if any(x in t_upper for x in ["SEMESTER PLAN", "PLAN OF STUDY"]):
                    skip_section = True
                    continue

                current = t
                is_ba = "B.A." in t_upper or "BACHELOR OF ARTS" in t_upper
                is_bs = "B.S." in t_upper or "BACHELOR OF SCIENCE" in t_upper
                
                if is_ba and "BS" in major_string:
                    skip_section = True
                elif is_bs and "BA" in major_string:
                    skip_section = True
                else:
                    skip_section = False 
            continue
            
        if skip_section:
            continue

        is_alt = 'orclass' in tag.get('class', [])
        
        links = tag.find_all("a")
        for a in links:
            code_raw = a.get_text(strip=True).upper().replace("\xa0", " ").replace("_", " ")
            if not re.search(r"[A-Z\s]+\d+", code_raw):
                continue
                
            clean_code = re.search(r"([A-Z\s]+\d+)", code_raw).group(1).strip()
            
            td = a.find_parent("td")
            if td and td.get_text(strip=True).lower().startswith("or"):
                is_alt = True
                
            if is_alt:
                clean_code = "OR " + clean_code
                
            tds = tag.find_all("td")
            title = ""
            for i, td_elem in enumerate(tds):
                if a in td_elem.find_all("a") and i + 1 < len(tds):
                    title = tds[i + 1].get_text(strip=True)
                    break
                    
            sections.setdefault(current, [])
            if not any(c[0] == clean_code for c in sections[current]):
                sections[current].append((clean_code, title, 3))

    return sections

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
        is_pick_list = any(k in sec_upper for k in ["FOCUS", "EXPERIENTIAL", "ELECTIVE", "CHOOSE", "SELECT"])
        
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
                results.append({"section": section, "code": first_code.replace("OR ", ""), "title": first_title, "status": status_text, "grade": "—"})

    return results

# ─────────────────────────────────────────────────────────────────
# AI TOOLS (LIVE SCRAPING & EXPORT)
# ─────────────────────────────────────────────────────────────────

def get_course_details(course_code: str) -> dict:
    """
    Fetches the live description, prerequisites, and credit hours for a given Mizzou course code.
    """
    clean_code = course_code.upper().strip()
    match = re.match(r"([A-Z\s]+)\s+(\d+\w*)", clean_code)
    if not match:
        return {"error": f"Invalid course code format: {clean_code}"}
        
    subject = match.group(1).strip()
    subject_url = subject.replace(" ", "_").lower()
    url = f"https://catalog.missouri.edu/courseofferings/{subject_url}/"
    
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        resp = requests.get(url, headers=headers, timeout=8)
        if resp.status_code != 200:
            return {"error": f"Could not find catalog page for {subject}."}
            
        soup = BeautifulSoup(resp.text, "html.parser")
        blocks = soup.find_all("div", class_="courseblock")
        search_str_1 = clean_code
        search_str_2 = clean_code.replace(" ", "_")
        
        for block in blocks:
            text = block.get_text(separator=" | ", strip=True)
            if search_str_1 in text.upper() or search_str_2 in text.upper():
                return {
                    "course": clean_code,
                    "catalog_data": text 
                }
                
        return {"error": f"Course {clean_code} not found on the live {subject} catalog page."}
    except Exception as e:
        return {"error": f"Failed to fetch course details: {str(e)}"}

def export_schedule(course_codes: list[str]) -> str:
    """
    Saves the student's agreed-upon upcoming semester schedule to a CSV file.
    """
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


# ─────────────────────────────────────────────────────────────────
# STREAMLIT WEB INTERFACE & AI CHAT
# ─────────────────────────────────────────────────────────────────

def init_chat_session(transcript_data, gap_analysis_results):
    """Initializes the Gemini chat and stores it in session state."""
    client = genai.Client(api_key=API_KEY)

    advisor_persona = """
    You are an expert academic advisor for the University of Missouri (Mizzou). 
    Your goal is to help students navigate their degree path based strictly on the gap analysis provided.
    
    CRITICAL RULES:
    1. Prerequisite Checking: NEVER recommend a course without calling 'get_course_details' to ensure the student has completed its prerequisites.
    2. Semester Balancing: A standard semester is 12-15 credits. Do not recommend more than 2 high-level technical classes in a single semester.
    3. Exporting: When the student agrees on a final list of classes for next semester, use the 'export_schedule' tool to save it for them.
    4. Keep your conversational responses concise. Ask one question at a time.
    """

    config = types.GenerateContentConfig(
        system_instruction=advisor_persona,
        tools=[get_course_details, export_schedule],
        temperature=0.2, 
    )

    student_context = f"""
    Here is the student's current academic profile:
    Name: {transcript_data['name']}
    Majors: {', '.join(transcript_data['majors'])}
    GPA: {transcript_data['gpa']}
    
    Here is their Degree Gap Analysis (JSON format):
    {json.dumps(gap_analysis_results, indent=2)}
    
    Review this data silently, then introduce yourself to the student and ask what term they are scheduling for.
    """

    chat = client.chats.create(model='gemini-2.5-flash', config=config)
    
    # ─── GRACEFUL ERROR HANDLING ADDED HERE ───
    try:
        initial_response = chat.send_message(student_context)
        return chat, initial_response.text
    except Exception as e:
        error_msg = f"⚠️ **API Error:** I couldn't initialize the chat. (Details: {e})"
        return chat, error_msg

def main():
    st.set_page_config(page_title="Mizzou AI Advisor", page_icon="🐯", layout="wide")
    st.title("🐯 Mizzou AI Academic Advisor")
    
    with st.sidebar:
        st.header("1. Upload Transcript")
        uploaded_file = st.file_uploader("Upload your Unofficial Transcript (PDF)", type="pdf")
        
        if st.button("Reset Session"):
            st.session_state.clear()
            st.rerun()

    if uploaded_file is not None:
        if "analyzed" not in st.session_state:
            with st.spinner("Parsing transcript and fetching live requirements..."):
                t = parse_transcript(uploaded_file)
                st.session_state.transcript = t
                
                all_results = {}
                if t["majors"]:
                    for major in t["majors"]:
                        reqs = get_requirements(major)
                        if reqs:
                            all_results[major] = gap_analysis(t["courses"], reqs)
                
                st.session_state.gap_analysis = all_results
                st.session_state.analyzed = True

                if API_KEY != "YOUR_GEMINI_API_KEY_HERE":
                    chat_session, first_msg = init_chat_session(t, all_results)
                    st.session_state.chat = chat_session
                    st.session_state.messages = [{"role": "assistant", "content": first_msg}]
                else:
                    st.error("Please add your Gemini API Key at the top of the script!")

        t = st.session_state.transcript
        with st.expander("📋 View Parsed Transcript Data"):
            st.write(f"**Name:** {t['name']}")
            st.write(f"**Major(s):** {', '.join(t['majors']) if t['majors'] else 'None detected'}")
            st.write(f"**GPA:** {t['gpa']} | **Credits:** {t['hours']}")

        st.divider()
        st.subheader("Chat with your AI Advisor")

        if "messages" in st.session_state:
            for msg in st.session_state.messages:
                with st.chat_message(msg["role"], avatar="🐯" if msg["role"] == "assistant" else "👤"):
                    st.markdown(msg["content"])

            if user_input := st.chat_input("Type your message here..."):
                with st.chat_message("user", avatar="👤"):
                    st.markdown(user_input)
                st.session_state.messages.append({"role": "user", "content": user_input})

                with st.chat_message("assistant", avatar="🐯"):
                    with st.spinner("Thinking..."):
                        try:
                            response = st.session_state.chat.send_message(user_input)
                            st.markdown(response.text)
                            st.session_state.messages.append({"role": "assistant", "content": response.text})
                        except Exception as e:
                            st.error(f"Error communicating with Gemini: {e}")

    else:
        st.info("👈 Please upload your transcript PDF in the sidebar to begin.")

if __name__ == "__main__":
    main()