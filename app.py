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
        # Add animated header
        st.markdown("""
        <style>
            @keyframes fadeIn {
                0% { opacity: 0; }
                100% { opacity: 1; }
            }
            .fade-in { animation: fadeIn 0.5s; }
        </style>
        """, unsafe_allow_html=True)

        # Custom avatars
        with st.chat_message("assistant", avatar="🏥"):
            st.markdown(f"**{self.settings['app']['title']}**")
            st.caption("AI Medical Assistant | Powered by Clinical Guidelines")

        # Example questions carousel
        with st.sidebar.expander("💡 Example Questions"):
            examples = [
                "Explain Type 2 diabetes management",
                "Latest hypertension treatment guidelines",
                "Side effects of metformin",
                "Pediatric asthma prevention strategies"
            ]
            for ex in examples:
                if st.button(ex, use_container_width=True):
                    st.session_state.ex_query = ex

        st.title(self.settings["app"]["title"])
        st.write(self.settings["app"]["description"])

        # Initialize components if not in session state
        if "chain_manager" not in st.session_state or st.session_state.chain_manager is None:
            with st.spinner("Initializing components..."):
                self.initialize_components()

        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Handle user input
        if prompt := st.chat_input("What would you like to know about your health?"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # Generate response using session state
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                try:
                    with st.spinner("Thinking..."):
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