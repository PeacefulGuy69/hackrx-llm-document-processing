from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Union
from enum import Enum


class QueryType(str, Enum):
    COVERAGE = "coverage"
    CLAIM = "claim"
    POLICY = "policy"
    GENERAL = "general"


class DocumentType(str, Enum):
    PDF = "pdf"
    DOCX = "docx"
    EMAIL = "email"
    TEXT = "text"


class StructuredQuery(BaseModel):
    """Structured representation of a parsed query"""
    age: Optional[int] = Field(None, description="Age of the person")
    gender: Optional[str] = Field(None, description="Gender (M/F/Male/Female)")
    procedure: Optional[str] = Field(None, description="Medical procedure or treatment")
    location: Optional[str] = Field(None, description="Location/city")
    policy_duration: Optional[str] = Field(None, description="Policy duration or age")
    policy_type: Optional[str] = Field(None, description="Type of policy")
    amount: Optional[float] = Field(None, description="Amount mentioned in query")
    query_type: QueryType = Field(QueryType.GENERAL, description="Type of query")
    original_query: str = Field(..., description="Original query text")
    extracted_entities: Dict[str, Any] = Field(default_factory=dict, description="Additional extracted entities")


class DocumentChunk(BaseModel):
    """Represents a chunk of processed document"""
    content: str = Field(..., description="Text content of the chunk")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Metadata about the chunk")
    embedding: Optional[List[float]] = Field(None, description="Vector embedding of the chunk")
    source: str = Field(..., description="Source document identifier")
    page_number: Optional[int] = Field(None, description="Page number if applicable")
    chunk_id: str = Field(..., description="Unique identifier for the chunk")


class RetrievedClause(BaseModel):
    """Represents a retrieved clause with similarity score"""
    content: str = Field(..., description="Content of the clause")
    source: str = Field(..., description="Source document")
    page_number: Optional[int] = Field(None, description="Page number")
    similarity_score: float = Field(..., description="Similarity score")
    clause_type: Optional[str] = Field(None, description="Type of clause")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class DecisionJustification(BaseModel):
    """Justification for a decision with referenced clauses"""
    decision: str = Field(..., description="The decision made")
    reasoning: str = Field(..., description="Explanation of the reasoning")
    referenced_clauses: List[RetrievedClause] = Field(..., description="Clauses used for decision")
    confidence_score: float = Field(..., description="Confidence in the decision (0-1)")


class ProcessingResult(BaseModel):
    """Final processing result"""
    decision: str = Field(..., description="Final decision (approved/rejected/pending)")
    amount: Optional[float] = Field(None, description="Amount if applicable")
    justification: DecisionJustification = Field(..., description="Detailed justification")
    structured_query: StructuredQuery = Field(..., description="Parsed query structure")
    processing_time: float = Field(..., description="Time taken to process in seconds")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional processing metadata")


class QueryRequest(BaseModel):
    """Request model for query processing"""
    documents: Union[str, List[str]] = Field(..., description="Document URL(s) or content")
    questions: List[str] = Field(..., description="List of questions to process")


class QueryResponse(BaseModel):
    """Response model for query processing"""
    answers: List[str] = Field(..., description="List of answers corresponding to questions")
    structured_responses: Optional[List[Dict[str, Any]]] = Field(None, description="Structured JSON responses for each question")


class StructuredDecisionResponse(BaseModel):
    """Structured decision response matching problem statement requirements"""
    decision: str = Field(..., description="Decision status: approved, rejected, or pending")
    amount: Optional[float] = Field(None, description="Amount if applicable") 
    justification: str = Field(..., description="Detailed justification")
    confidence: float = Field(..., description="Confidence score (0-1)")
    referenced_clauses: List[str] = Field(..., description="List of referenced clause summaries")
    processing_metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional processing information")


class DocumentProcessingRequest(BaseModel):
    """Request for document processing"""
    document_url: str = Field(..., description="URL of the document to process")
    document_type: Optional[DocumentType] = Field(None, description="Type of document")


class DocumentProcessingResponse(BaseModel):
    """Response for document processing"""
    document_id: str = Field(..., description="Unique identifier for processed document")
    chunks_created: int = Field(..., description="Number of chunks created")
    processing_time: float = Field(..., description="Processing time in seconds")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Processing metadata")


class HealthCheckResponse(BaseModel):
    """Health check response"""
    status: str = Field(..., description="Service status")
    version: str = Field(..., description="API version")
    timestamp: str = Field(..., description="Current timestamp")


class ErrorResponse(BaseModel):
    """Error response model"""
    error: str = Field(..., description="Error message")
    details: Optional[str] = Field(None, description="Additional error details")
    error_code: Optional[str] = Field(None, description="Error code")
