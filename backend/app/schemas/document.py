from pydantic import BaseModel
from datetime import datetime
from app.models.document import DocumentStatus


class DocumentBase(BaseModel):
    filename: str
    original_filename: str
    file_type: str
    file_size: int


class DocumentCreate(DocumentBase):
    workspace_id: int


class DocumentResponse(DocumentBase):
    id: int
    workspace_id: int
    status: DocumentStatus
    chunk_count: int
    error_message: str | None
    created_at: datetime
    updated_at: datetime
    # NexusRAG fields
    page_count: int = 0
    image_count: int = 0
    table_count: int = 0
    parser_version: str | None = None
    processing_time_ms: int = 0
    custom_metadata: dict | None = None

    model_config = {"from_attributes": True}


class DocumentUploadResponse(BaseModel):
    id: int
    filename: str
    status: DocumentStatus
    message: str
