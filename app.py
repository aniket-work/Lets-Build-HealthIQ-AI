# app.py
import streamlit as st
from pathlib import Path
import  numpy as np
from src.constants import SETTINGS_PATH, CONFIG_PATH
from src.utils import load_yaml_config, load_json_config, setup_environment
from src.session_manager import initialize_components


class MedicalChatbotUI:
    def __init__(self):
        """Initialize the Medical Chatbot UI."""
        self.settings = load_yaml_config(SETTINGS_PATH)
        self.config = load_json_config(CONFIG_PATH)
        setup_environment(self.config["api_keys"]["huggingface"])

        if "messages" not in st.session_state:
            st.session_state.messages = []

    def initialize_components(self):
        """Initialize components and store them in session state."""
        try:
            st.info("Starting initialization...")
            # Store components in session state
            (
                st.session_state.doc_processor,
                st.session_state.embeddings_manager,
                st.session_state.llm_manager,
                st.session_state.chain_manager,
                st.session_state.vectorstore
            ) = initialize_components(self.settings, self.config)
            st.success("All components initialized successfully!")
        except Exception as e:
            st.error(f"Error initializing components: {str(e)}")
            raise

    # Add to MedicalChatbotUI.render()
    def render(self):
        """Render the Streamlit UI with a professional healthcare design."""
        # Custom CSS for professional healthcare look
        st.markdown("""
        <style>
            /* Main container */
            .stApp {
                background: linear-gradient(135deg, #f0f2f5, #e6f4f1);
                font-family: 'Arial', sans-serif;
            }

            /* Chat container */
            .stChatMessage {
                border-radius: 15px;
                padding: 15px;
                margin: 10px 0;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            }

            /* User message */
            .stChatMessage.user {
                background: #0078d4;
                color: white;
                margin-left: 20%;
            }

            /* Assistant message */
            .stChatMessage.assistant {
                background: #ffffff;
                color: #333;
                margin-right: 20%;
                border: 1px solid #ddd;
            }

            /* Sidebar */
            .stSidebar {
                background: #ffffff;
                padding: 20px;
                border-radius: 10px;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            }

            /* Buttons */
            .stButton button {
                background: #0078d4;
                color: white;
                border-radius: 8px;
                padding: 10px 20px;
                font-size: 14px;
                transition: background 0.3s;
            }

            .stButton button:hover {
                background: #005bb5;
            }

            /* Input box */
            .stTextInput input {
                border-radius: 8px;
                padding: 10px;
                border: 1px solid #ddd;
            }

            /* Header */
            .stMarkdown h1 {
                color: #0078d4;
                font-size: 28px;
                font-weight: bold;
            }

            /* Example questions */
            .example-question {
                background: #f8f9fa;
                border-radius: 8px;
                padding: 10px;
                margin: 5px 0;
                cursor: pointer;
                transition: background 0.3s;
            }

            .example-question:hover {
                background: #e9ecef;
            }
        </style>
        """, unsafe_allow_html=True)

        # Header with logo and tagline
        st.markdown("""
        <div style="text-align: center; padding: 20px;">
            <h1 style="color: #0078d4; font-size: 36px; font-weight: bold;">
                🏥 HealthIQ AI
            </h1>
            <p style="color: #555; font-size: 16px;">
                Your AI-powered medical assistant for accurate, evidence-based healthcare guidance.
            </p>
        </div>
        """, unsafe_allow_html=True)

        # Sidebar with model information and controls
        with st.sidebar:
            st.markdown("""
            <div style="text-align: center;">
                <h3 style="color: #0078d4;">Model Information</h3>
                <p style="color: #555;">Using model: <strong>{}</strong></p>
            </div>
            """.format(self.settings['model']['llm']['name']), unsafe_allow_html=True)

            if st.button("Reset Model", key="reset_model"):
                if "llm_manager" in st.session_state:
                    st.session_state.llm_manager.reset_model()
                    st.success("Model reset successfully!")

            st.markdown("---")

            # Example questions
            st.markdown("""
            <div style="text-align: center;">
                <h4 style="color: #0078d4;">💡 Example Questions</h4>
            </div>
            """, unsafe_allow_html=True)

            examples = [
                "Explain Type 2 diabetes management",
                "Latest hypertension treatment guidelines",
                "Side effects of metformin",
                "Pediatric asthma prevention strategies"
            ]

            for ex in examples:
                if st.button(ex, key=ex, use_container_width=True):
                    st.session_state.messages.append({"role": "user", "content": ex})
                    st.session_state.messages.append({"role": "assistant", "content": "Thinking..."})

        # Initialize components if not already done
        if "chain_manager" not in st.session_state or st.session_state.chain_manager is None:
            with st.spinner("Initializing components..."):
                self.initialize_components()

        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"], avatar="🧑‍⚕️" if message["role"] == "assistant" else "👤"):
                st.markdown(message["content"])

        # Chat input
        if prompt := st.chat_input("What would you like to know about your health?"):
            # Add user message
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user", avatar="👤"):
                st.markdown(prompt)

            # Generate response
            with st.chat_message("assistant", avatar="🧑‍⚕️"):
                message_placeholder = st.empty()
                try:
                    with st.spinner("Analyzing your query..."):
                        response = st.session_state.chain_manager.get_response(prompt)
                        message_placeholder.markdown(response)
                        st.session_state.messages.append({"role": "assistant", "content": response})
                except Exception as e:
                    st.error(f"Error generating response: {str(e)}")


def main():
    """Main entry point for the Streamlit application."""
    st.set_page_config(
        page_title="HealthIQ AI, Power of Vertical AI Agent",
        page_icon="🏥",
        layout="wide"
    )

    # Initialize and render the UI
    app = MedicalChatbotUI()
    app.render()


if __name__ == "__main__":
    main()