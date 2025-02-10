# src/session_manager.py
import streamlit as st
from typing import Optional, Tuple, Any
from pathlib import Path
import tempfile
import os
import time

from core.document_loader import DocumentProcessor
from core.embeddings import EmbeddingsManager
from core.llm import LLMManager
from core.chain import ChainManager
from langchain.vectorstores import Chroma


def get_persist_directory():
    """Get or create a persistent directory for vector store."""
    if 'persist_dir' not in st.session_state:
        # Create a persistent directory in /tmp
        persist_dir = tempfile.mkdtemp(prefix='vectorstore_')
        st.session_state.persist_dir = persist_dir
    return st.session_state.persist_dir


def init_session_state():
    """Initialize session state variables if they don't exist."""
    if "initialized" not in st.session_state:
        st.session_state.initialized = False
    if "doc_processor" not in st.session_state:
        st.session_state.doc_processor = None
    if "embeddings_manager" not in st.session_state:
        st.session_state.embeddings_manager = None
    if "llm_manager" not in st.session_state:
        st.session_state.llm_manager = None
    if "chain_manager" not in st.session_state:
        st.session_state.chain_manager = None
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None
    if "embeddings_data" not in st.session_state:
        st.session_state.embeddings_data = None


def initialize_components(settings: dict, config: dict) -> Tuple[
    DocumentProcessor, EmbeddingsManager, LLMManager, ChainManager, Chroma]:
    """Initialize all components if not already initialized."""
    init_session_state()

    if not st.session_state.initialized:
        try:
            # Document processing
            st.session_state.doc_processor = DocumentProcessor(
                chunk_size=settings["chunking"]["chunk_size"],
                chunk_overlap=settings["chunking"]["chunk_overlap"]
            )

            # Process documents
            pdf_path = Path(settings["paths"]["pdf_directory"])
            documents = st.session_state.doc_processor.process_documents(pdf_path)

            # Setup embeddings
            st.session_state.embeddings_manager = EmbeddingsManager(
                model_name=settings["model"]["embeddings"]["name"]
            )

            # Get persistent directory
            persist_dir = get_persist_directory()

            # Create vectorstore with persistence
            st.session_state.vectorstore = st.session_state.embeddings_manager.create_vectorstore(
                documents=documents,
                persist_dir=persist_dir
            )

            # Add delay to ensure vectorstore is ready
            time.sleep(2)

            # Get embeddings data with retry logic
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    data = st.session_state.embeddings_manager.get_all_embeddings(
                        st.session_state.vectorstore
                    )
                    st.session_state.embeddings_data = data
                    break
                except Exception as e:
                    if attempt < max_retries - 1:
                        time.sleep(2)
                        continue
                    raise

            # Setup retriever
            retriever = st.session_state.embeddings_manager.get_retriever(
                st.session_state.vectorstore,
                k=settings["retriever"]["search_k"]
            )

            # Setup LLM
            st.session_state.llm_manager = LLMManager(
                model_name=settings["model"]["llm"]["name"],
                temperature=settings["model"]["llm"]["temperature"],
                max_tokens=settings["model"]["llm"]["max_tokens"],
                top_p=settings["model"]["llm"]["top_p"],
                base_url=settings["model"]["llm"]["base_url"]
            )

            # Setup chain
            st.session_state.chain_manager = ChainManager(
                retriever=retriever,
                llm=st.session_state.llm_manager.llm,
                prompt_template=config["prompt_template"]
            )

            st.session_state.initialized = True

        except Exception as e:
            st.error(f"Error initializing components: {str(e)}")
            raise

    return (
        st.session_state.doc_processor,
        st.session_state.embeddings_manager,
        st.session_state.llm_manager,
        st.session_state.chain_manager,
        st.session_state.vectorstore
    )


def get_components() -> Tuple[Optional[Any], ...]:
    """Get initialized components from session state."""
    return (
        st.session_state.get("doc_processor"),
        st.session_state.get("embeddings_manager"),
        st.session_state.get("llm_manager"),
        st.session_state.get("chain_manager"),
        st.session_state.get("vectorstore")
    )


def get_embeddings_data() -> Optional[dict]:
    """Get stored embeddings data from session state."""
    return st.session_state.get("embeddings_data")