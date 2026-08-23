import os
from typing import TypedDict, List
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langgraph.graph import StateGraph, END


import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI

# Safe key retrieval across Streamlit Cloud and Azure Container Apps
api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")

if not api_key:
    try:
        api_key = st.secrets.get("GOOGLE_API_KEY")
    except Exception:
        pass

llm = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    google_api_key=api_key
)


class ChatState(TypedDict, total=False):
    messages: List[BaseMessage]
    sub_topic: str
    turn_count: int
    is_final_turn: bool


def get_context(sub_topic: str, user_query: str) -> str:
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
        results = db.similarity_search(user_query, k=3, filter={"sub_topic": sub_topic})
        return "\n\n".join([doc.page_content for doc in results]) if results else ""
    except Exception:
        return "No specific syllabus context found."


def socratic_tutor(state: ChatState) -> dict:
    sub_topic = state.get("sub_topic", "OCR A-Level Computer Science")
    user_query = state["messages"][-1].content if state.get("messages") else ""
    context = get_context(sub_topic, user_query)

    system_prompt = f"""You are an expert Socratic OCR A-Level Computer Science Tutor.
    Topic Focus: {sub_topic}
    Syllabus Context:
    {context}

    Guide the student step-by-step using probing questions and constructive hints. Never give away full answers directly."""

    #llm = ChatGoogleGenerativeAI(model="gemini-3.6-flash")
    messages_to_send = [HumanMessage(content=system_prompt)] + list(state["messages"])
    response = llm.invoke(messages_to_send)

    return {
        "messages": state["messages"] + [response]
    }


def didactic_fallback(state: ChatState) -> dict:
    sub_topic = state.get("sub_topic", "")
    user_query = state["messages"][-1].content if state.get("messages") else ""
    context = get_context(sub_topic, user_query)

    system_prompt = f"""You are a strict Cambridge OCR A-Level Computer Science Senior Examiner wrapping up a Socratic revision session.
    Topic Focus: {sub_topic}
    Syllabus Context:
    {context}

    STRICT FORMATTING & LATEX RULES:
    - NEVER use LaTeX math delimiters like $, $$, \\(, or \\). Write all complexity, matrices, and variables in plain text or Markdown bold/code (e.g., O(V^2), O(1), 1000 x 1000).

    STRICT OCR MARKING RUBRIC:
    - 80–100%: Accurate concepts AND precise OCR specification keywords used consistently throughout.
    - 50–79%: Conceptually sound, but relies on informal/layperson language instead of required exam terms.
    - Below 50%: Vague explanations, partial misconceptions, or missing core terminology.

    INSTRUCTIONS FOR SESSION ENDING:
    1. **Validate Final Answer:** Directly validate the student's final input in detail first. If code/pseudocode was discussed, include the full exam-standard corrected solution.
    2. **Performance Assessment:** Apply the rubric above, list technical terms used well vs. missed across the dialogue, give 1–2 targeted feedback points, and recommend next sub-topics.
    3. **Structured Summary Note:** Provide a clean, comprehensive topic summary data drop.

    FORMAT YOUR OUTPUT EXACTLY AS FOLLOWS (Include the exact separator string ===SPLIT=== on its own line):

    [Your validation of the student's final answer and corrected solution]

    ### 📊 Session Performance
    - **Overall Accuracy:** [X]%
    - **Targeted Feedback:** 
      - [Constructive feedback point 1]
      - [Constructive feedback point 2]
    - **Exam Terminology:**
      - **Keywords Used Well:** [Term 1, Term 2]
      - **Missed Terms to Learn:** [Term 3, Term 4]
    - **Recommended Next Revision Topic(s):** [Sub-topic 1 / Sub-topic 2]

    ===SPLIT===

    ### 💡 Topic Summary: [Topic Name]
    [Detailed structured summary data drop in plain text/Markdown only, NO $ symbols]

    CRITICAL RULE: DO NOT ask any follow-up questions anywhere in your response. Conclude cleanly."""

    #llm = ChatGoogleGenerativeAI(model="gemini-3.6-flash")
    messages_to_send = [HumanMessage(content=system_prompt)] + list(state["messages"])
    response = llm.invoke(messages_to_send)

    return {
        "messages": state["messages"] + [response]
    }


def route_turn(state: ChatState) -> str:
    # Guarantee fallback routing on Turn 7
    if state.get("is_final_turn") or state.get("turn_count", 0) >= 7:
        return "didactic_fallback"

    messages = state.get("messages", [])
    human_count = 0
    for m in messages:
        if getattr(m, "type", "") == "human" or "Human" in type(m).__name__ or (isinstance(m, dict) and m.get("role") in ["student", "user", "human"]):
            human_count += 1

    if human_count >= 7:
        return "didactic_fallback"

    return "socratic_tutor"


builder = StateGraph(ChatState)
builder.add_node("socratic_tutor", socratic_tutor)
builder.add_node("didactic_fallback", didactic_fallback)

builder.set_conditional_entry_point(
    route_turn,
    {
        "socratic_tutor": "socratic_tutor",
        "didactic_fallback": "didactic_fallback",
    }
)

builder.add_edge("socratic_tutor", END)
builder.add_edge("didactic_fallback", END)

workflow = builder.compile()