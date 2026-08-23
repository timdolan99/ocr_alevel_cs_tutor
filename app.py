import streamlit as st
import re
import importlib
import socratic_fsm
importlib.reload(socratic_fsm)
from socratic_fsm import workflow
from langchain_core.messages import HumanMessage, AIMessage

import os
from langchain_google_genai import ChatGoogleGenerativeAI

# Checks Streamlit Secrets first, then falls back to Azure Environment Variables
api_key = st.secrets.get("GOOGLE_API_KEY") or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")

llm = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    google_api_key=api_key
)

# --- Helper 1: Text Extractor ---
def extract_clean_text(response) -> str:
    """Extracts plain text response from Gemini output structures."""
    if isinstance(response, str):
        return response
    if hasattr(response, "content"):
        return extract_clean_text(response.content)
    if isinstance(response, list) and len(response) > 0:
        first_item = response[0]
        if isinstance(first_item, dict):
            return first_item.get("text", str(first_item))
        elif hasattr(first_item, "text"):
            return first_item.text
        return extract_clean_text(first_item)
    if isinstance(response, dict):
        if "text" in response:
            return response["text"]
        elif "content" in response:
            return extract_clean_text(response["content"])
    return str(response)

# --- Helper 2: LaTeX Math Cleaner ---
def clean_latex(text: str) -> str:
    """Removes stray LaTeX dollar signs and converts math expressions to clean bold text."""
    text = re.sub(r'\$([^\$]+)\$', r'<b>\1</b>', text)
    return text.replace('$', '')

# --- Helper 3: Markdown to HTML Converter for Custom Div Bubbles ---
def md_to_html(text: str) -> str:
    """Converts Markdown elements into HTML with compact line gaps for mobile/desktop layouts."""
    text = re.sub(r'^####\s+(.*$)', r'<h4 style="margin: 4px 0 1px 0; font-size: 1.05em; color: inherit;">\1</h4>', text, flags=re.MULTILINE)
    text = re.sub(r'^###\s+(.*$)', r'<h3 style="margin: 6px 0 2px 0; font-size: 1.1em; color: inherit;">\1</h3>', text, flags=re.MULTILINE)
    text = re.sub(r'^##\s+(.*$)', r'<h2 style="margin: 8px 0 2px 0; font-size: 1.2em; color: inherit;">\1</h2>', text, flags=re.MULTILINE)
    text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', text)
    text = re.sub(r'\*(.*?)\*', r'<i>\1</i>', text)
    text = re.sub(r'^\s*[-*]\s+(.*$)', r'<div style="margin: 1px 0;">• \1</div>', text, flags=re.MULTILINE)
    
    # Strip newlines following block tags (divs and headings) so they don't turn into <br> tags
    text = re.sub(r'(</(div|h2|h3|h4)>)\s*\n+', r'\1', text)
    text = re.sub(r'\n{2,}', '<br>', text)
    text = text.replace('\n', '<br>')
    
    # Collapse any stacked consecutive <br> tags into a single line break
    return re.sub(r'(<br\s*/?>\s*)+', '<br>', text)

# --- 1. Custom Messaging App CSS ---
st.markdown("""
    <style>
    /* Background & Main Container */
    .stApp { 
        background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%); 
    }
    div[data-testid="stSidebar"] { 
        background-color: #ffffff; 
        border-right: 1px solid #e2e8f0;
    }

    /* Custom Green Progress Bar */
    div[data-testid="stProgress"] > div > div > div {
        background-color: #22c55e !important;
    }

    h1, h2, h3 { 
        color: #0f172a; 
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        font-weight: 700;
    }
    
    /* Sleek Vibrant CS Header */
    .chat-header { 
        background: linear-gradient(135deg, #0f172a 0%, #2563eb 100%); 
        color: white; 
        padding: 22px; 
        font-weight: 700; 
        text-align: center; 
        font-size: 1.3em; 
        border-radius: 16px; 
        box-shadow: 0 10px 15px -3px rgba(37, 99, 235, 0.25);
        margin-bottom: 24px; 
        letter-spacing: 0.5px;
    }

    /* Target Selection Card Container */
    .selection-card {
        background: #ffffff;
        padding: 24px;
        border-radius: 16px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03);
        border: 1px solid #e2e8f0;
        margin-bottom: 20px;
    }

    /* Bipolar Message Bubbles */
    .tutor-msg { 
        background-color: #ffffff; 
        color: #1e293b; 
        padding: 14px 18px; 
        border-radius: 18px 18px 18px 4px; 
        margin-bottom: 12px; 
        max-width: 82%; 
        line-height: 1.35; 
        box-shadow: 0 2px 4px rgba(0,0,0,0.04);
        border: 1px solid #e2e8f0;
    }
    .student-msg { 
        background: linear-gradient(135deg, #1d4ed8 0%, #3b82f6 100%); 
        color: white; 
        padding: 14px 18px; 
        border-radius: 18px 18px 4px 18px; 
        margin-bottom: 12px; 
        max-width: 82%; 
        margin-left: auto; 
        line-height: 1.35;
        box-shadow: 0 4px 6px -1px rgba(29, 78, 216, 0.2);
    }
    .summary-box { 
        background: #fefce8; 
        border-left: 5px solid #eab308; 
        padding: 10px 14px; 
        border-radius: 12px; 
        color: #713f12; 
        font-size: 0.93em; 
        margin: 8px 0;
        max-width: 85%; 
        line-height: 1.3; 
        box-shadow: 0 2px 4px rgba(0,0,0,0.03);
    }

    /* Button Styling */
    .stButton > button {
        border-radius: 12px !important;
        font-weight: 600 !important;
        transition: all 0.2s ease-in-out !important;
    }
    </style>
""", unsafe_allow_html=True)

# --- 2. OCR A-Level Computer Science Specification Structure ---
OCR_CS_TOPICS = {
    "Component 01: Computer Systems": [
        "1.1.1 Structure & Function of the Processor",
        "1.1.2 Types of Processor",
        "1.1.3 Input, Output and Storage",
        "1.2.1 Operating Systems & Systems Software",
        "1.2.2 Applications Generation & Translators",
        "1.2.3 Software Development Lifecycles",
        "1.2.4 Types of Programming Language & Assembly",
        "1.2.5 Object-Oriented Programming",
        "1.3.1 Compression, Encryption and Hashing",
        "1.3.2 Databases & SQL",
        "1.3.3 Networks & Protocols",
        "1.3.4 Web Technologies",
        "1.4.1 Data Types & Binary Representation",
        "1.4.2 Data Structures",
        "1.4.3 Boolean Algebra & Logic Gates",
        "1.5.1 Computing Related Legislation",
        "1.5.2 Moral, Ethical, Social & Cultural Issues"
    ],
    "Component 02: Algorithms and Programming": [
        "2.1 Elements of Computational Thinking",
        "2.2.1 Programming Techniques & Recursion",
        "2.2.2 Computational Methods",
        "2.3.1 Algorithmic Complexity & Big O",
        "2.3.1 Data Structure Algorithms & Traversals",
        "2.3.1 Standard Searching & Sorting Algorithms"
    ]
}

# --- 3. Initialize Session States ---
if "active_unit" not in st.session_state:
    st.session_state.active_unit = None
if "active_topic" not in st.session_state:
    st.session_state.active_topic = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "graph_state" not in st.session_state:
    st.session_state.graph_state = {
        "messages": [],
        "sub_topic": None,
        "turn_count": 0,
        "is_final_turn": False
    }

def reset_session():
    st.session_state.active_unit = None
    st.session_state.active_topic = None
    st.session_state.messages = []
    st.session_state.graph_state = {
        "messages": [],
        "sub_topic": None,
        "turn_count": 0,
        "is_final_turn": False
    }
    st.rerun()

# --- 4. Single-Screen View Router ---
if st.session_state.active_topic is None:
    st.markdown('<div class="chat-header">💻 OCR A-Level Computer Science Socratic Coach</div>', unsafe_allow_html=True)
    st.markdown('<div class="selection-card">', unsafe_allow_html=True)
    st.subheader("🎯 Select Revision Target")
    st.write("Choose a component unit and subtopic below to start your guided practice session:")
    
    selected_unit = st.selectbox("📘 Step 1: Choose Component:", options=list(OCR_CS_TOPICS.keys()))
    selected_subtopic = st.selectbox("🔍 Step 2: Choose Specific CS Topic:", options=OCR_CS_TOPICS[selected_unit])
    
    st.write("")
    if st.button("🚀 Start Socratic Session", type="primary", use_container_width=True):
        st.session_state.active_unit = selected_unit
        st.session_state.active_topic = selected_subtopic
        st.session_state.graph_state["sub_topic"] = selected_subtopic
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

else:
    topic_code = st.session_state.active_topic.split(" ")[0]
    st.markdown(f'<div class="chat-header">🎓 Computer Science Coach ({topic_code})</div>', unsafe_allow_html=True)
    
    student_turns = sum(1 for m in st.session_state.messages if m.get("role") == "student")

    with st.sidebar:
        st.subheader("📌 Active Target")
        st.info(f"**Component:** {st.session_state.active_unit}\n\n**Topic:** {st.session_state.active_topic}")
        
        st.metric(label="Turn Counter", value=f"{student_turns} / 7")
        st.progress(min(student_turns / 7, 1.0))
        
        st.write("---")
        if st.button("🔄 New Session / Change Topic", use_container_width=True):
            reset_session()

    # Initial Greeting
    if len(st.session_state.messages) == 0:
        initial_greeting = (
            f"Welcome! We're exploring **{st.session_state.active_topic}** today. "
            f"To get us started, can you suggest a keyword or concept within this topic "
            f"that you would like to revise?"
        )
        st.session_state.messages.append({"role": "tutor", "content": initial_greeting, "style": "tutor-msg"})
        st.session_state.graph_state["messages"].append(AIMessage(content=initial_greeting))

    # Render Chat Messages
    for msg in st.session_state.messages:
        html_content = md_to_html(msg["content"])
        if msg["role"] == "tutor":
            div_class = msg.get("style", "tutor-msg")
            header = "💡 <b>Summary Note</b>" if div_class == "summary-box" else "💻 <b>CS Tutor</b>"
            st.markdown(f'<div class="{div_class}">{header}<br><br>{html_content}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="student-msg">🎒 <b>Student</b><br><br>{html_content}</div>', unsafe_allow_html=True)

    # Session Status Alert
    if student_turns >= 7:
        st.info("🎉 **Session Complete!** You have completed all 7 turns of the A-Level Socratic dialogue and received your performance assessment and gold summary card.")

    # Permanent Bottom Chat Input Widget
    is_disabled = student_turns >= 7
    placeholder = "Session complete. Select a new topic in the sidebar to continue." if is_disabled else "Type your response here..."
    
    if user_input := st.chat_input(placeholder, disabled=is_disabled):
        st.session_state.messages.append({"role": "student", "content": user_input})
        st.session_state.graph_state["messages"].append(HumanMessage(content=user_input))
        
        current_student_turns = sum(1 for m in st.session_state.messages if m.get("role") == "student")
        st.session_state.graph_state["turn_count"] = current_student_turns
        st.session_state.graph_state["is_final_turn"] = (current_student_turns >= 7)

        with st.spinner("Analyzing response and generating assessment..."):
            updated_state = workflow.invoke(st.session_state.graph_state)
        
        last_msg = updated_state["messages"][-1]
        ai_reply = extract_clean_text(last_msg)
        ai_reply = clean_latex(ai_reply)

        # Robust Regex Splitting for Turn 7
        split_match = re.split(r'={3,}\s*SPLIT\s*={3,}', ai_reply, flags=re.IGNORECASE)
        if len(split_match) > 1:
            feedback_part = split_match[0].strip()
            summary_part = split_match[1].strip()
            
            st.session_state.messages.append({"role": "tutor", "content": feedback_part, "style": "tutor-msg"})
            st.session_state.messages.append({"role": "tutor", "content": summary_part, "style": "summary-box"})
        else:
            st.session_state.messages.append({"role": "tutor", "content": ai_reply, "style": "tutor-msg"})

        st.session_state.graph_state = updated_state
        st.rerun()