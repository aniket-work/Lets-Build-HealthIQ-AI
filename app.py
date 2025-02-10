# app.py
import streamlit as st
from pathlib import Path
from typing import Optional

from src.constants import SETTINGS_PATH, CONFIG_PATH
from src.utils import load_yaml_config, load_json_config, setup_environment
from core.document_loader import DocumentProcessor
from core.embeddings import EmbeddingsManager
from core.llm import LLMManager
from core.chain import ChainManager

# app.py
import streamlit as st
from pathlib import Path

from src.constants import SETTINGS_PATH, CONFIG_PATH
from src.utils import load_yaml_config, load_json_config, setup_environment
from src.session_manager import initialize_components


class MedicalChatbotUI:
    def __init__(self):
        """Initialize the Medical Chatbot UI."""
        self.settings = load_yaml_config(SETTINGS_PATH)
        self.config = load_json_config(CONFIG_PATH)

        # Setup environment
        setup_environment(self.config["api_keys"]["huggingface"])

        # Setup UI state
        if "messages" not in st.session_state:
            st.session_state.messages = []

    def initialize_components(self):
        """Initialize all required components using session manager."""
        try:
            st.info("Starting initialization...")

            # Initialize all components using session manager
            self.doc_processor, self.embeddings_manager, self.llm_manager, \
                self.chain_manager, self.vectorstore = initialize_components(self.settings, self.config)

            st.success("All components initialized successfully!")
        except Exception as e:
            st.error(f"Error initializing components: {str(e)}")
            raise

    def render(self):
        """Render the Streamlit UI."""
        st.title(self.settings["app"]["title"])
        st.write(self.settings["app"]["description"])

        # Add sidebar with model information
        with st.sidebar:
            st.title("Model Information")
            st.write(f"Using model: {self.settings['model']['llm']['name']}")
            if st.button("Reset Model"):
                if self.llm_manager:
                    self.llm_manager.reset_model()
                    st.success("Model reset successfully!")
            st.markdown("---")
            if st.button("Vector Space"):
                st.switch_page("pages/02_vector_space.py")

        # Initialize components if not already done
        if "chain_manager" not in st.session_state or st.session_state.chain_manager is None:
            with st.spinner("Initializing components..."):
                self.initialize_components()

        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Chat input
        if prompt := st.chat_input("What would you like to know about your health?"):
            # Add user message
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # Generate response
            with st.chat_message("assistant"):
                message_placeholder = st.empty()

                # Stream the response
                try:
                    with st.spinner("Thinking..."):
                        response = self.chain_manager.get_response(prompt)
                        message_placeholder.markdown(response)
                        st.session_state.messages.append({"role": "assistant", "content": response})
                except Exception as e:
                    st.error(f"Error generating response: {str(e)}")


def main():
    """Main entry point for the Streamlit application."""
    st.set_page_config(
        page_title="Medical Chatbot",
        page_icon="🏥",
        layout="wide"
    )

    # Initialize and render the UI
    app = MedicalChatbotUI()
    app.render()


if __name__ == "__main__":
    main()