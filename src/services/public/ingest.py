import inngest
from fastapi import status
from src import schemas
from src.services.internal import process_documents, dense_encode, sparse_encode
from src.repo.qdrant import upsert_nodes


def ingest_documents(ctx: inngest.Context) -> schemas.IngestionResponse:
    try:
        request = schemas.IngestRequest.model_validate(ctx.event.data)
        if not request.file_paths and not request.file_dir:
            raise ValueError("No file paths or directory provided in event data.")

        ctx.logger.info(
            f"Starting documents ingestion process to the collection '{request.collection_name}'."
        )

        nodes = process_documents(
            file_paths=request.file_paths, file_dir=request.file_dir
        )

        if len(nodes) == 0:
            raise ValueError("No nodes were created from the provided documents.")

        ctx.logger.info(f"Processed {len(nodes)} chunks with UUIDs and metadata.")

        # Create dense embeddings for the nodes
        dense_embeddings = dense_encode(
            texts=[node.text for node in nodes],
            titles=[node.metadata.get("title", "none") for node in nodes],
            text_type="document",
        )

        if len(dense_embeddings) != len(nodes):
            raise ValueError(
                f"Embeddings generation failed or returned incorrect count: {len(dense_embeddings)}"
            )

        ctx.logger.info(
            f"Generated {len(nodes)} dense embeddings with each embedding's size is: {len(dense_embeddings[0])}"
        )

        # Attach embeddings to nodes
        for node, embedding in zip(nodes, dense_embeddings):
            node.embedding = embedding

        # Create sparse embeddings for the nodes
        sparse_embeddings, vocab = sparse_encode(
            texts=[node.text for node in nodes],
            word_process_method="lemmatize",
        )

        print(f"Generated sparse embeddings shape: {sparse_embeddings.shape}")

        # Upsert nodes
        upsert_nodes(nodes=nodes, collection_name=request.collection_name)
        ctx.logger.info(
            f"Upserted {len(nodes)} nodes from {len(request.file_paths)} documents to Qdrant collection '{request.collection_name}'."
        )

        return schemas.IngestionResponse(
            status=status.HTTP_201_CREATED,
            message=f"Successfully ingested {len(nodes)} nodes into collection '{request.collection_name}'.",
        )

    except Exception as e:
        ctx.logger.error(f"Error while ingesting documents: {e}")

        return schemas.IngestionResponse(
            status=status.HTTP_500_INTERNAL_SERVER_ERROR, message=str(e)
        )
