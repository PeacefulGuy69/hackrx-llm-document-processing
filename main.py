from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging
import time
import os
from datetime import datetime

from models import (
    QueryRequest, QueryResponse, HealthCheckResponse, 
    ErrorResponse, DocumentProcessingRequest, DocumentProcessingResponse
)
from llm_processor import llm_processor
from config import config

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Security
security = HTTPBearer()


def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify the bearer token"""
    if credentials.credentials != config.BEARER_TOKEN:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication token"
        )
    return credentials


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    logger.info("Starting LLM Document Processing System...")
    
    # Startup
    try:
        # Perform health check
        health_status = await llm_processor.health_check()
        if health_status["status"] != "healthy":
            logger.warning(f"System started with warnings: {health_status}")
        else:
            logger.info("System started successfully")
    except Exception as e:
        logger.error(f"Error during startup: {str(e)}")
    
    yield
    
    # Shutdown
    logger.info("Shutting down LLM Document Processing System...")


# Create FastAPI app
app = FastAPI(
    title=config.APP_NAME,
    description=config.APP_DESCRIPTION,
    version=config.APP_VERSION,
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", response_model=HealthCheckResponse)
async def root():
    """Root endpoint"""
    return HealthCheckResponse(
        status="running",
        version=config.APP_VERSION,
        timestamp=datetime.now().isoformat()
    )


@app.get("/health", response_model=HealthCheckResponse)
async def health_check():
    """Health check endpoint"""
    try:
        health_status = await llm_processor.health_check()
        return HealthCheckResponse(
            status=health_status["status"],
            version=config.APP_VERSION,
            timestamp=datetime.now().isoformat()
        )
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Health check failed: {str(e)}"
        )


@app.post("/hackrx/run", response_model=QueryResponse)
async def process_hackrx_queries(
    request: QueryRequest,
    credentials: HTTPAuthorizationCredentials = Depends(verify_token)
):
    """
    Main endpoint for processing HackRx queries.
    
    This endpoint processes documents and answers questions based on their content.
    """
    try:
        start_time = time.time()
        
        # Validate request
        if not request.questions:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No questions provided"
            )
        
        if not request.documents:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No documents provided"
            )
        
        logger.info(f"Processing {len(request.questions)} questions for documents: {request.documents}")
        
        # Process the request
        response = await llm_processor.process_query_request(request)
        
        processing_time = time.time() - start_time
        logger.info(f"Request processed successfully in {processing_time:.2f}s")
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing queries: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing queries: {str(e)}"
        )


@app.post(f"{config.API_PREFIX}/document/process", response_model=DocumentProcessingResponse)
async def process_document(
    request: DocumentProcessingRequest,
    credentials: HTTPAuthorizationCredentials = Depends(verify_token)
):
    """Process a single document and add it to the search index"""
    try:
        start_time = time.time()
        
        logger.info(f"Processing document: {request.document_url}")
        
        # Process document
        chunks = await llm_processor.document_processor.process_document_from_url(
            request.document_url, 
            request.document_type
        )
        
        # Add to vector store
        await llm_processor.semantic_searcher.add_documents(chunks)
        
        # Cache the document
        llm_processor.document_cache.set(request.document_url, chunks)
        
        processing_time = time.time() - start_time
        
        response = DocumentProcessingResponse(
            document_id=request.document_url,
            chunks_created=len(chunks),
            processing_time=processing_time,
            metadata={
                "document_type": request.document_type,
                "source_url": request.document_url
            }
        )
        
        logger.info(f"Document processed: {len(chunks)} chunks in {processing_time:.2f}s")
        return response
        
    except Exception as e:
        logger.error(f"Error processing document: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing document: {str(e)}"
        )


@app.get(f"{config.API_PREFIX}/stats")
async def get_system_stats(
    credentials: HTTPAuthorizationCredentials = Depends(verify_token)
):
    """Get system statistics"""
    try:
        stats = await llm_processor.get_system_stats()
        return stats
    except Exception as e:
        logger.error(f"Error getting stats: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting stats: {str(e)}"
        )


@app.post(f"{config.API_PREFIX}/cache/clear")
async def clear_cache(
    credentials: HTTPAuthorizationCredentials = Depends(verify_token)
):
    """Clear all cached data"""
    try:
        await llm_processor.clear_cache()
        return {"message": "Cache cleared successfully"}
    except Exception as e:
        logger.error(f"Error clearing cache: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error clearing cache: {str(e)}"
        )


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler"""
    logger.error(f"Unhandled exception: {str(exc)}")
    return HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail="Internal server error"
    )


if __name__ == "__main__":
    import uvicorn
    
    # Get port from environment variable for cloud deployment compatibility
    port = int(os.getenv("PORT", config.PORT))
    host = os.getenv("HOST", config.HOST)
    
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=config.DEBUG,
        log_level="info"
    )
