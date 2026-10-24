"""
src/vectordb/index_builder.py

Builds a ChromaDB vector collection using precomputed E5 embeddings.

Pipeline

EmbeddingGenerator
        │
        ▼
embedding records
        │
        ▼
IndexBuilder
        │
        ▼
ChromaDB
"""

from __future__ import annotations

import logging
from typing import Any

from src.vectordb.chroma_manager import ChromaManager

logger = logging.getLogger(__name__)


class IndexBuilder:
    """
    Stores embedding vectors inside ChromaDB.

    Input records must come directly from
    EmbeddingGenerator.generate().
    """

    def __init__(
        self,
        db_path: str = r"D:\\inu\\QA_FINTECH_FINAL\\data\\test_chroma_db",
        collection_name: str = "fintech_rag",
    ):

        self.manager = ChromaManager(
            persist_directory=db_path
        )

        self.collection = self.manager.get_or_create_collection(
            collection_name
        )

        logger.info(
            "Collection loaded: %s",
            collection_name,
        )

    def build_index(
        self,
        embedding_records: list[dict[str, Any]],
    ) -> None:
        """
        Stores vectors inside ChromaDB.
        """

        if not embedding_records:
            raise ValueError("No embedding records supplied.")

        ids = []
        documents = []
        embeddings = []
        metadatas = []

        for record in embedding_records:

            ids.append(
                record["chunk_id"]
            )

            documents.append(
                record["text_preview"]
            )

            embeddings.append(
                record["embedding"]
            )

            metadata = {

                "document_id":
                    record["document_id"],

                "chunk_index":
                    record["chunk_index"],

                "token_count":
                    record["chunk_metadata"]["token_count"],

                "complexity":
                    record["chunk_metadata"]["complexity_level"],

                "language":
                    record["chunk_metadata"]["language"],

                "domain":
                    record["chunk_metadata"]["domain"]

            }

            metadatas.append(metadata)

        self.collection.upsert(

            ids=ids,

            documents=documents,

            embeddings=embeddings,

            metadatas=metadatas,

        )

        logger.info(
            "Indexed %d chunks.",
            len(ids),
        )


if __name__ == "__main__":

    import json

    logging.basicConfig(level=logging.INFO)

    builder = IndexBuilder()

    with open(
        r"D:\\inu\\QA_FINTECH_FINAL\\data\\embeddings\\chunk_embeddings.json",
        "r",
        encoding="utf-8",
    ) as f:

        records = json.load(f)

    builder.build_index(records)

    print()

    print("Collection:", builder.collection.name)

    print("Documents:", builder.collection.count())