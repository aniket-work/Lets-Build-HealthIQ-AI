# core/chroma_validator.py
import logging
import chromadb
from chromadb.config import Settings
import numpy as np
from typing import List, Dict, Any, Optional
import time


class ChromaValidator:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)

    def init_client(self, persist_dir: str) -> chromadb.PersistentClient:
        """Initialize Chroma client with explicit settings."""
        try:
            settings = Settings(
                anonymized_telemetry=False,
                is_persistent=True,
                persist_directory=persist_dir
            )

            client = chromadb.PersistentClient(
                path=persist_dir,
                settings=settings
            )

            self.logger.info(f"ChromaDB client initialized with persist_dir: {persist_dir}")
            return client

        except Exception as e:
            self.logger.error(f"Failed to initialize ChromaDB client: {str(e)}")
            raise

    def validate_or_create_collection(self, client: chromadb.PersistentClient,
                                      collection_name: str) -> chromadb.Collection:
        """Validate existing collection or create new one."""
        try:
            collections = client.list_collections()
            collection_exists = any(c.name == collection_name for c in collections)

            if collection_exists:
                self.logger.info(f"Found existing collection: {collection_name}")
                return client.get_collection(collection_name)

            self.logger.info(f"Creating new collection: {collection_name}")
            collection = client.create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"}
            )

            if not collection:
                raise ValueError("Failed to create collection")

            time.sleep(1)  # Allow collection to initialize
            return collection

        except Exception as e:
            self.logger.error(f"Error in collection validation/creation: {str(e)}")
            raise

    def add_documents_to_collection(self,
                                    collection: chromadb.Collection,
                                    documents: List[str],
                                    embeddings: List[List[float]],
                                    metadatas: Optional[List[Dict]] = None) -> bool:
        """Add documents to collection with validation."""
        try:
            # Generate unique IDs
            ids = [f"doc_{i}_{int(time.time())}" for i in range(len(documents))]

            # Add documents in batches
            batch_size = 50
            for i in range(0, len(documents), batch_size):
                end_idx = min(i + batch_size, len(documents))

                # Prepare batch data
                batch_docs = documents[i:end_idx]
                batch_embeddings = embeddings[i:end_idx]
                batch_ids = ids[i:end_idx]
                batch_metadata = metadatas[i:end_idx] if metadatas else None

                # Add batch
                collection.add(
                    documents=batch_docs,
                    embeddings=batch_embeddings,
                    metadatas=batch_metadata,
                    ids=batch_ids
                )

                self.logger.info(f"Added batch of {len(batch_docs)} documents")
                time.sleep(0.5)  # Prevent overwhelming the database

            # Verify final count
            time.sleep(1)  # Allow collection to settle
            final_count = collection.count()
            expected_count = len(documents)

            if final_count == expected_count:
                self.logger.info(f"Successfully added all {final_count} documents")
                return True
            else:
                self.logger.warning(f"Document count mismatch. Expected: {expected_count}, Got: {final_count}")
                return False

        except Exception as e:
            self.logger.error(f"Error adding documents to collection: {str(e)}")
            return False