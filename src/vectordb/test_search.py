from src.vectordb.chroma_manager import ChromaManager
from src.embedding.embedding_model import EmbeddingModel


DB_PATH = r"D:\\inu\\QA_FINTECH_FINAL\\data\\test_chroma_db"


COLLECTION_NAME = "fintech_rag"


manager = ChromaManager(
    persist_directory=DB_PATH
)


collection = manager.get_or_create_collection(
    COLLECTION_NAME
)


print(
    "Documents:",
    collection.count()
)


# -----------------------------
# Load E5 embedding model
# -----------------------------

model = EmbeddingModel()


query = "CBE Birr ምንድን ነው?"


query_embedding = model.encode_batch(
    [
        "query: " + query
    ]
)[0]


# -----------------------------
# Chroma dense search
# -----------------------------

results = collection.query(

    query_embeddings=[
        query_embedding
    ],

    n_results=3,

    include=[
        "documents",
        "metadatas",
        "distances"
    ]

)


for i in range(len(results["ids"][0])):

    print("\n----------------")

    print(
        "ID:",
        results["ids"][0][i]
    )

    print(
        "Distance:",
        results["distances"][0][i]
    )

    print(
        "TEXT:",
        results["documents"][0][i][:300]
    )