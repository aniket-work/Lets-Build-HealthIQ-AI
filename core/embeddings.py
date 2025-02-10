# core/embeddings.py
from typing import List, Dict, Any
from langchain_community.vectorstores import Chroma  # Updated import
from langchain_community.embeddings import OllamaEmbeddings  # Updated import
from langchain.docstore.document import Document
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import time
import logging
from core.chroma_validator import ChromaValidator


class EmbeddingsManager:
    def __init__(self, model_name: str = "nomic-embed-text"):
        """Initialize embeddings manager with Ollama model."""
        self.embeddings = OllamaEmbeddings(
            model=model_name,
            base_url="http://localhost:11434"
        )
        self.validator = ChromaValidator()
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)

    def get_retriever(self, vectorstore: Chroma, k: int = 4):
        """
        Get retriever from vector store.

        Args:
            vectorstore: Chroma vector store instance
            k: Number of documents to retrieve

        Returns:
            BaseRetriever: Configured retriever
        """
        try:
            return vectorstore.as_retriever(
                search_kwargs={"k": k}
            )
        except Exception as e:
            self.logger.error(f"Error creating retriever: {str(e)}")
            raise

    def find_similar_vectors(self, query_vector: np.ndarray, document_vectors: np.ndarray, k: int = 5) -> np.ndarray:
        """
        Find the k most similar vectors to a query vector using cosine similarity.

        Args:
            query_vector: The query embedding vector to compare against
            document_vectors: Matrix of document embedding vectors to search through
            k: Number of similar vectors to return (default: 5)

        Returns:
            np.ndarray: Indices of the k most similar vectors
        """
        try:
            self.logger.info(f"Finding {k} most similar vectors")

            # Ensure vectors are 2D arrays
            if len(query_vector.shape) == 1:
                query_vector = query_vector.reshape(1, -1)
            if len(document_vectors.shape) == 1:
                document_vectors = document_vectors.reshape(1, -1)

            # Normalize vectors
            query_norm = np.linalg.norm(query_vector, axis=1, keepdims=True)
            doc_norm = np.linalg.norm(document_vectors, axis=1, keepdims=True)

            # Avoid division by zero
            query_norm = np.where(query_norm == 0, 1e-10, query_norm)
            doc_norm = np.where(doc_norm == 0, 1e-10, doc_norm)

            query_vector_normalized = query_vector / query_norm
            document_vectors_normalized = document_vectors / doc_norm

            # Calculate cosine similarities
            similarities = np.dot(query_vector_normalized, document_vectors_normalized.T).flatten()

            # Get indices of top k similar vectors
            top_k_indices = np.argsort(similarities)[-k:][::-1]

            return top_k_indices

        except Exception as e:
            self.logger.error(f"Error finding similar vectors: {str(e)}")
            raise

    def get_query_embedding(self, query: str) -> np.ndarray:
        """
        Get embedding for a search query.

        Args:
            query: Search query text

        Returns:
            np.ndarray: Query embedding vector
        """
        try:
            embedding = self.embeddings.embed_query(query)
            return np.array([embedding])  # Return as 2D array for consistency
        except Exception as e:
            self.logger.error(f"Error getting query embedding: {str(e)}")
            raise
    def get_all_embeddings(self, vectorstore: Chroma) -> dict:
        """
        Get all embeddings and their metadata from the vector store.

        Args:
            vectorstore: Chroma vector store instance

        Returns:
            dict: Dictionary containing embeddings, documents, and metadata
        """
        try:
            self.logger.info("Retrieving all embeddings from vector store")

            # Get collection
            collection = vectorstore._collection

            # Get all embeddings
            result = collection.get(
                include=['embeddings', 'documents', 'metadatas']
            )

            if not result or 'embeddings' not in result:
                raise ValueError("No embeddings found in vector store")

            return {
                'embeddings': result['embeddings'],
                'documents': result['documents'],
                'metadata': result['metadatas']
            }

        except Exception as e:
            self.logger.error(f"Error retrieving embeddings: {str(e)}")
            raise

    def create_vectorstore(self, documents: List[Document], persist_dir: str) -> Chroma:
        """Create a vector store from the provided documents."""
        if not documents:
            raise ValueError("No documents provided for creating vector store")

        try:
            self.logger.info(f"Creating vectorstore for {len(documents)} documents")

            # Initialize Chroma client
            client = self.validator.init_client(persist_dir)

            # Pre-compute embeddings in batches
            self.logger.info("Pre-computing embeddings...")
            texts = [doc.page_content for doc in documents]
            metadata = [doc.metadata for doc in documents]

            # Process in smaller batches to prevent memory issues
            batch_size = 10
            all_embeddings = []

            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                self.logger.info(f"Processing batch {i // batch_size + 1}")
                batch_embeddings = self.embeddings.embed_documents(batch_texts)
                all_embeddings.extend(batch_embeddings)
                time.sleep(0.5)  # Prevent overwhelming the embedding service

            # Create or get collection
            collection = self.validator.validate_or_create_collection(client, "medical_docs")

            # Add documents in batches
            success = self.validator.add_documents_to_collection(
                collection=collection,
                documents=texts,
                embeddings=all_embeddings,
                metadatas=metadata
            )

            if not success:
                raise ValueError("Failed to add documents to collection")

            # Allow collection to settle
            time.sleep(1)

            # Verify the collection has the expected count
            count = collection.count()
            if count != len(documents):
                raise ValueError(f"Document count mismatch. Expected: {len(documents)}, Got: {count}")

            # Create vector store instance
            vectorstore = Chroma(
                client=client,
                collection_name="medical_docs",
                embedding_function=self.embeddings,
                persist_directory=persist_dir
            )

            self.logger.info(f"Successfully created vector store with {count} documents")
            return vectorstore

        except Exception as e:
            self.logger.error(f"Error in create_vectorstore: {str(e)}")
            raise