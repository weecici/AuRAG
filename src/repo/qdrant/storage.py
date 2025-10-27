from functools import lru_cache
from qdrant_client import QdrantClient
from llama_index.core.schema import BaseNode
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import models
from src.core import config


@lru_cache(maxsize=1)
def get_qdrant_client() -> QdrantClient:
    return QdrantClient(url=config.QDRANT_URL, timeout=30)


def ensure_collection_exists(
    collection_name: str, vector_size: int = config.EMBEDDING_DIM
) -> None:
    client = get_qdrant_client()

    collections = client.get_collections().collections
    if any(col.name == collection_name for col in collections):
        print(f"Collection '{collection_name}' already exists.")
        return

    client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(
            size=vector_size,
            distance=models.Distance.COSINE,
            hnsw_config=models.HnswConfigDiff(
                m=32,
                ef_construct=100,
            ),
        ),
    )
    print(f"Created collection '{collection_name}' with HNSW index for ANN search.")


def get_vector_store(collection_name: str) -> QdrantVectorStore:
    client = get_qdrant_client()
    ensure_collection_exists(collection_name)
    return QdrantVectorStore(client=client, collection_name=collection_name)


def upsert_nodes(nodes: list[BaseNode], collection_name: str) -> None:
    if not nodes:
        raise ValueError("No nodes provided for upserting")

    if not all(node.embedding is not None for node in nodes):
        raise ValueError("All nodes must have embeddings attached before upserting")

    vector_store = get_vector_store(collection_name)

    vector_store.add(nodes)

    print(
        f"Successfully upserted {len(nodes)} nodes into collection '{collection_name}'."
    )
