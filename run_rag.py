"""
run_rag.py

Amharic FinTech RAG Pipeline

QueryAnalyzer
      |
RetrievalPipeline
      |
DenseRetriever + SparseRetriever
      |
Hybrid Retrieval
      |
PromptBuilder
      |
Ollama Qwen2.5:7b
"""


from pathlib import Path
import json
import requests
import chromadb


# ============================================================
# Imports
# ============================================================

from src.embedding.embedding_model import EmbeddingModel

from src.morphology.query_analyzer import QueryAnalyzer

from src.retrieval.dense_retriever import DenseRetriever

from src.retrieval.sparse_retriever import SparseRetriever

from src.retrieval.retrieval_pipeline import (
    RetrievalPipeline,
    QueryAnalyzerMorphologyAdapter
)

from src.generation.prompt_builder import PromptBuilder



# ============================================================
# Paths
# ============================================================

BASE_DIR = Path(
    r"D:\inu\QA_FINTECH_FINAL"
)


CHROMA_PATH = (
    BASE_DIR /
    "data" /
    "test_chroma_db"
)


CHUNK_FILE = (
    BASE_DIR /
    "data" /
    "processed" /
    "chunked_documents.json"
)


COLLECTION_NAME = "cbe_fintech_chunks"



# ============================================================
# Load Embedding Model
# ============================================================


print(
    "\nLoading multilingual-e5-large..."
)


embedding_model = EmbeddingModel()



def embedding_function(
    texts: list[str]
):
    """
    Query embedding only.

    IMPORTANT:
    Return complete matrix.
    Do NOT use [0].
    """

    return embedding_model.encode_batch(
        texts
    )



# ============================================================
# Load Chroma
# ============================================================


def load_chroma():

    print(
        "\nLoading ChromaDB..."
    )


    client = chromadb.PersistentClient(
        path=str(CHROMA_PATH)
    )


    collections = client.list_collections()


    names = [
        c.name
        for c in collections
    ]


    print(
        "Existing collections:",
        names
    )


    if COLLECTION_NAME not in names:

        raise RuntimeError(
            f"Collection {COLLECTION_NAME} not found"
        )


    collection = client.get_collection(
        name=COLLECTION_NAME
    )


    print(
        "Loaded:",
        collection.name
    )


    return collection



# ============================================================
# Ollama
# ============================================================


def ask_ollama(prompt):


    response = requests.post(

        "http://localhost:11434/api/generate",

        json={

            "model":
                "qwen2.5:7b",

            "prompt":
                prompt,

            "stream":
                False

        },

        timeout=300

    )


    response.raise_for_status()


    return response.json()["response"]




# ============================================================
# Main
# ============================================================


def main():



    # ---------------------------
    # Chroma
    # ---------------------------

    collection = load_chroma()



    # ---------------------------
    # Dense Retrieval
    # ---------------------------

    dense = DenseRetriever(

        chroma_collection=collection,

        embedding_fn=embedding_function

    )



    # ---------------------------
    # Sparse Retrieval
    # ---------------------------

    morphology_adapter = (
        QueryAnalyzerMorphologyAdapter()
    )


    sparse = SparseRetriever(

        chunked_documents_path=CHUNK_FILE,

        morphological_analyzer=morphology_adapter

    )



    # ---------------------------
    # Retrieval Pipeline
    # ---------------------------


    pipeline = RetrievalPipeline(

        dense_retriever=dense,

        sparse_retriever=sparse,

        alpha=0.5

    )



    # ---------------------------
    # Analyzer
    # ---------------------------


    analyzer = QueryAnalyzer()



    # ---------------------------
    # Query
    # ---------------------------


    query = input(
        "\nAsk Amharic FinTech question: "
    )



    print(
        "\n===== QUERY ANALYSIS ====="
    )


    analysis = analyzer.analyze(
        query
    )


    print(
        json.dumps(
            analysis,
            ensure_ascii=False,
            indent=2
        )
    )



    # ---------------------------
    # Retrieval
    # ---------------------------


    print(
        "\n===== RETRIEVAL ====="
    )


    result = pipeline.retrieve(
        query
    )



    print(
        "Strategy:",
        result.strategy
    )



    for i,chunk in enumerate(
        result.chunks,
        start=1
    ):

        print(
            f"""
[{i}]
ID:
{chunk.chunk_id}

{chunk.text[:400]}
"""
        )



    # ---------------------------
    # Prompt
    # ---------------------------


    builder = PromptBuilder()


    prompt_result = builder.build(
        result
    )


    messages = prompt_result.messages


    final_prompt = messages[-1]["content"]



    # ---------------------------
    # Generation
    # ---------------------------


    print(
        "\n===== QWEN2.5 ====="
    )


    answer = ask_ollama(
        final_prompt
    )


    print(answer)




if __name__ == "__main__":

    main()