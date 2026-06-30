import streamlit as st
import streamlit.components.v1 as components
import os
import uuid

def load_chatbot_component(title: str = "🧠 Gemini Chatbot", chatbot_html_path: str = "assets/chatbot.html"):
    """
    Renders the floating Gemini chatbot in any Streamlit app.
    
    Args:
        title (str): Optional title for the page or chatbot section.
        chatbot_html_path (str): Path to the chatbot.html file.
    
    Usage:
        from chatbot_component import load_chatbot_component
        load_chatbot_component()
    """

    # Set page configuration (only once)
    if "chatbot_loaded" not in st.session_state:
        st.set_page_config(page_title="Floating Gemini Chatbot", layout="wide")
        st.session_state["chatbot_loaded"] = True

    # Assign a persistent session_id for current Streamlit session
    if "session_id" not in st.session_state:
        st.session_state["session_id"] = str(uuid.uuid4())

    

    # Ensure HTML file exists
    html_path = os.path.abspath(chatbot_html_path)
    if not os.path.exists(html_path):
        st.error(f"❌ Chatbot HTML file not found: {html_path}")
        return

    # Load and inject session_id dynamically
    with open(html_path, "r", encoding="utf-8") as f:
        chatbot_html = f.read()

    # Inject Streamlit session ID into HTML JavaScript context
    chatbot_html = chatbot_html.replace(
        "sessionId = sessionStorage.getItem(\"sessionId\")",
        f'sessionId = "{st.session_state["session_id"]}"'
    )

    components.html(chatbot_html, height=6, scrolling=False)

    st.success("✅ Gemini Chatbot Loaded Successfully", icon="💬")
