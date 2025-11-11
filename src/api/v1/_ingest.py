import inngest
import src.services.public as public_svcs
from fastapi import APIRouter
from src import schemas
from src.core import inngest_client

router = APIRouter()


@inngest_client.create_function(
    fn_id="ingest-documents",
    trigger=inngest.TriggerEvent(event="rag/ingest-documents"),
    retries=0,
)
async def ingest_documents(ctx: inngest.Context) -> dict[str, any]:
    request = schemas.IngestionRequest.model_validate(ctx.event.data)
    return public_svcs.ingest_documents(request).model_dump()


@router.post(
    "/ingest",
    response_model=schemas.IngestionResponse,
    summary="Start document ingestion",
    description="Triggers an asynchronous job to ingest documents from the specified file paths or directory.",
)
async def ingest(request: schemas.IngestionRequest) -> schemas.IngestionResponse:
    return public_svcs.ingest_documents(request)
