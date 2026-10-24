# =====================================================================
# File Path: src/vectordb/chroma_manager.py
# =====================================================================

from __future__ import annotations
import logging
import os
import chromadb
from chromadb.config import Settings
from typing import Optional
from typing import Optional, Any
# Setup clean structural log tracing
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("ChromaManager")


class ChromaManager:
    """
    Encapsulates life-cycle connection pooling and directory states 
    for the vector database persistent storage.
    """
    def __init__(self, persist_directory: str = "D:\inu\QA_FINTECH_FINAL\data\test_chroma_db"):
        self.persist_directory = persist_directory
        self._client: Optional[chromadb.ClientAPI] = None
        self._initialize_client()

    def _initialize_client(self) -> None:
        """Instantiates a thread-safe, localized persistent DB storage environment."""
        try:
            # Ensure workspace path structure exists cleanly before boot
            os.makedirs(self.persist_directory, exist_ok=True)
            
            self._client = chromadb.PersistentClient(
                path=self.persist_directory,
                settings=Settings(anonymized_telemetry=False, allow_reset=True)
            )
            logger.info(f"Database engine successfully bounded to filesystem path: {self.persist_directory}")
        except Exception as e:
            logger.error(f"Failed to bind persistent client: {str(e)}")
            raise e

    @property
    def client(self) -> chromadb.ClientAPI:
        """Thread-safe property accessor for client instance operations."""
        if self._client is None:
            self._initialize_client()
        return self._client

    def get_or_create_collection(self, collection_name: str, embedding_function: Optional[object] = None) -> Any:
        """Retrieves an existing vector index bucket or allocates a new tracking matrix."""
        try:
            collection = self.client.get_or_create_collection(
                name=collection_name,
                embedding_function=embedding_function,
                metadata={"hnsw:space": "cosine"}  # Establish structural similarity mapping rules
            )
            logger.info(f"Storage target collection '{collection_name}' fetched/verified.")
            return collection
        except Exception as e:
            logger.error(f"Critical allocation failure for collection {collection_name}: {str(e)}")
            raise e


if __name__ == "__main__":
    print("\n=== EXECUTING STANDALONE COHORT VALIDATION: CHROMA_MANAGER ===\n")
    
    # Initialize the engine segment using an explicit test verification namespace
    test_directory = r"D:\\inu\\QA_FINTECH_FINAL\\data\\test_chroma_db"
    manager = ChromaManager(persist_directory=test_directory)
    
    # Trigger collection generation lifecycle
    target_collection = manager.get_or_create_collection(collection_name="test_fintech_verification")
    
    print("\n--- IN-MEMORY PROPERTY METADATA CHECK ---")
    print(f"Target Collection Name Identity: {target_collection.name}")
    print(f"Target Collection Metadata Config: {target_collection.metadata}")
    print(f"Target Local Storage Directory Presence: {os.path.exists(test_directory)}")