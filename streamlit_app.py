import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(
    page_title="EMIPredict-AI",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Hide Streamlit's own UI
st.markdown(
    """
    <style>
        #MainMenu {
            visibility: hidden;
        }

        header {
            visibility: hidden;
        }

        footer {
            visibility: hidden;
        }

        .stApp {
            margin: 0;
            padding: 0;
        }

        .block-container {
            padding: 0 !important;
            margin: 0 !important;
            max-width: 100% !important;
        }

        iframe {
            width: 100% !important;
            border: none !important;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

components.iframe(
    "https://emipredict-ai-frontend.onrender.com",
    height=1000,
    scrolling=True,
)