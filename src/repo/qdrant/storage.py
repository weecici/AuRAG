from functools import lru_cache
from qdrant_client import QdrantClient, models
from llama_index.core.schema import BaseNode
from src.core import config
from scipy.sparse import csr_matrix
from typing import List


@lru_cache(maxsize=1)
def get_qdrant_client() -> QdrantClient:
    return QdrantClient(url=config.QDRANT_URL, timeout=30)


def ensure_collection_exists(
    collection_name: str,
    dense_name: str = config.EMBEDDING_MODEL,
    sparse_name: str = config.SPARSE_MODEL,
    vector_size: int = config.EMBEDDING_DIM,
) -> None:
    client = get_qdrant_client()

    collections = client.get_collections().collections
    if any(col.name == collection_name for col in collections):
        print(f"Collection '{collection_name}' already exists.")
        return

    client.create_collection(
        collection_name=collection_name,
        vectors_config={
            dense_name: models.VectorParams(
                size=vector_size,
                distance=models.Distance.COSINE,
                hnsw_config=models.HnswConfigDiff(
                    m=32,
                    ef_construct=100,
                ),
            ),
        },
        sparse_vectors_config={
            sparse_name: models.SparseVectorParams(),
        },
    )
    print(f"Created collection '{collection_name}'")


def upsert_data(
    nodes: list[BaseNode],
    dense_embeddings: list[list[float]],
    sparse_embeddings: csr_matrix,
    vocab: dict[str, int],
    collection_name: str,
    dense_name: str = config.EMBEDDING_MODEL,
    sparse_name: str = config.SPARSE_MODEL,
    vector_size: int = config.EMBEDDING_DIM,
) -> None:
    if not nodes:
        raise ValueError("No nodes provided for upserting")

    if len(dense_embeddings) != len(nodes):
        raise ValueError("All nodes must have embeddings attached before upserting")

    if sparse_embeddings.shape != (len(nodes), len(vocab)):
        raise ValueError(
            f"Sparse embeddings shape {sparse_embeddings.shape} does not match expected shape ({len(nodes)}, {len(vocab)})"
        )

    client = get_qdrant_client()
    ensure_collection_exists(
        collection_name=collection_name,
        dense_name=dense_name,
        sparse_name=sparse_name,
        vector_size=vector_size,
    )

    # Convert each CSR row to Qdrant SparseVector
    def sparse_vectorize(i: int) -> models.SparseVector:
        start = sparse_embeddings.indptr[i]
        end = sparse_embeddings.indptr[i + 1]
        indices: List[int] = sparse_embeddings.indices[start:end].tolist()
        values: List[float] = sparse_embeddings.data[start:end].tolist()

        # Qdrant accepts empty sparse vectors, but keep explicit empty lists
        return models.SparseVector(indices=indices, values=values)

    points: list[models.PointStruct] = [
        models.PointStruct(
            id=0,
            vector={
                dense_name: [0.0] * vector_size,
                sparse_name: models.SparseVector(indices=[], values=[]),
            },
            payload={
                "vocab": vocab,
            },
        )
    ]

    for i, node in enumerate(nodes):
        vector_map: dict[str, object] = {
            dense_name: dense_embeddings[i],
            sparse_name: sparse_vectorize(i),
        }

        points.append(
            models.PointStruct(
                id=node.id_,
                vector=vector_map,
                payload={
                    "text": node.text,
                    "metadata": node.metadata,
                },
            )
        )

    out = client.upsert(
        collection_name=collection_name,
        points=points,
    )

    # Qdrant returns UpdateStatus.ACKNOWLEDGED or COMPLETED
    if out.status not in (
        models.UpdateStatus.ACKNOWLEDGED,
        models.UpdateStatus.COMPLETED,
    ):
        raise RuntimeError(f"Failed to upsert nodes: {out}")
