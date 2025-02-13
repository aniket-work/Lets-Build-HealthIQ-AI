# document_loader.py
from pathlib import Path
from typing import List
import logging
from langchain.docstore.document import Document
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os


class DocumentProcessor:
    def __init__(self, chunk_size: int = 300, chunk_overlap: int = 50):
        """Initialize document processor with chunking parameters."""
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)

    def write_chunks_to_file(self, chunks: List[Document], output_file: str = "debug_chunks.txt"):
        """Write all document chunks to a file for debugging."""
        with open(output_file, 'w', encoding='utf-8') as f:
            for i, chunk in enumerate(chunks):
                f.write(f"\n\n=== CHUNK {i} ===\n")
                f.write(f"Source: {chunk.metadata.get('source', 'unknown')}\n")
                f.write(f"Page: {chunk.metadata.get('page', 'unknown')}\n")
                f.write(f"Content:\n{chunk.page_content}\n")
                f.write(f"Document ID: {chunk.metadata.get('doc_id', 'N/A')}\n")
                f.write(f"Publication Date: {chunk.metadata.get('pub_date', 'N/A')}\n")

    def load_documents(self, pdf_directory: Path) -> List[Document]:
        """Load PDF documents from the specified directory."""
        try:
            # Convert to absolute path and verify existence
            pdf_directory = pdf_directory.resolve()
            self.logger.info(f"Looking for PDFs in: {pdf_directory}")

            if not pdf_directory.exists():
                raise ValueError(f"PDF directory does not exist: {pdf_directory}")

            # List all PDF files in directory
            pdf_files = list(pdf_directory.glob("*.pdf"))
            self.logger.info(f"Found PDF files: {[f.name for f in pdf_files]}")

            if not pdf_files:
                raise ValueError(f"No PDF files found in directory: {pdf_directory}")

            # Load documents
            loader = PyPDFDirectoryLoader(str(pdf_directory))
            documents = loader.load()

            self.logger.info(f"Loaded {len(documents)} document pages")
            if not documents:
                raise ValueError(f"No content loaded from PDFs in: {pdf_directory}")

            return documents

        except Exception as e:
            self.logger.error(f"Error loading documents: {str(e)}")
            raise

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into chunks."""
        if not documents:
            raise ValueError("No documents provided for splitting")

        try:
            split_docs = self.text_splitter.split_documents(documents)
            self.logger.info(f"Split into {len(split_docs)} chunks")

            if not split_docs:
                raise ValueError("No content found after splitting documents")

            return split_docs

        except Exception as e:
            self.logger.error(f"Error splitting documents: {str(e)}")
            raise

    def process_documents(self, pdf_directory: Path) -> List[Document]:
        """Load and process documents in one go."""
        try:
            self.logger.info(f"Starting document processing from: {pdf_directory}")

            # Load documents
            documents = self.load_documents(pdf_directory)
            self.logger.info(f"Loaded {len(documents)} documents")

            # Split into chunks
            split_docs = self.split_documents(documents)
            self.logger.info(f"Split into {len(split_docs)} chunks")

            # Write chunks for debugging
            self.write_chunks_to_file(split_docs)

            if not split_docs:
                raise ValueError("No documents produced after processing")

            return split_docs

        except Exception as e:
            self.logger.error(f"Error processing documents: {str(e)}")
            raise