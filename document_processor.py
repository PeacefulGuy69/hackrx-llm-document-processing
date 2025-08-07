import asyncio
import logging
from typing import List, Dict, Any, Optional, Union
import aiohttp
import tempfile
import os
from pathlib import Path

import fitz  # PyMuPDF
import docx
from email.mime.text import MimeText
from email.parser import Parser
import tiktoken

from models import DocumentChunk, DocumentType
from config import config

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DocumentProcessor:
    """Handles document processing and text extraction"""
    
    def __init__(self):
        self.encoding = tiktoken.encoding_for_model("gpt-4")
        self.max_chunk_size = config.MAX_CHUNK_SIZE
        self.chunk_overlap = config.CHUNK_OVERLAP
    
    async def process_document_from_url(self, url: str, document_type: Optional[DocumentType] = None) -> List[DocumentChunk]:
        """Process document from URL and return chunks"""
        try:
            # Download document
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status != 200:
                        raise Exception(f"Failed to download document: {response.status}")
                    
                    content = await response.read()
                    
                    # Create temporary file
                    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                        temp_file.write(content)
                        temp_path = temp_file.name
            
            # Detect document type if not provided
            if not document_type:
                document_type = self._detect_document_type(url, content)
            
            # Extract text based on document type
            text_content = await self._extract_text(temp_path, document_type)
            
            # Clean up temporary file
            os.unlink(temp_path)
            
            # Create chunks
            chunks = self._create_chunks(text_content, url)
            
            logger.info(f"Processed document from {url}: {len(chunks)} chunks created")
            return chunks
            
        except Exception as e:
            logger.error(f"Error processing document from {url}: {str(e)}")
            raise
    
    def _detect_document_type(self, url: str, content: bytes) -> DocumentType:
        """Detect document type from URL and content"""
        url_lower = url.lower()
        
        if url_lower.endswith('.pdf') or content.startswith(b'%PDF'):
            return DocumentType.PDF
        elif url_lower.endswith(('.docx', '.doc')) or content.startswith(b'PK'):
            return DocumentType.DOCX
        elif 'email' in url_lower or content.startswith(b'From:'):
            return DocumentType.EMAIL
        else:
            return DocumentType.TEXT
    
    async def _extract_text(self, file_path: str, document_type: DocumentType) -> str:
        """Extract text from document based on type"""
        try:
            if document_type == DocumentType.PDF:
                return self._extract_pdf_text(file_path)
            elif document_type == DocumentType.DOCX:
                return self._extract_docx_text(file_path)
            elif document_type == DocumentType.EMAIL:
                return self._extract_email_text(file_path)
            else:
                return self._extract_text_file(file_path)
        except Exception as e:
            logger.error(f"Error extracting text from {file_path}: {str(e)}")
            raise
    
    def _extract_pdf_text(self, file_path: str) -> str:
        """Extract text from PDF file"""
        text_content = []
        
        with fitz.open(file_path) as doc:
            for page_num, page in enumerate(doc):
                text = page.get_text()
                if text.strip():
                    text_content.append(f"Page {page_num + 1}:\n{text}")
        
        return "\n\n".join(text_content)
    
    def _extract_docx_text(self, file_path: str) -> str:
        """Extract text from DOCX file"""
        doc = docx.Document(file_path)
        text_content = []
        
        for paragraph in doc.paragraphs:
            if paragraph.text.strip():
                text_content.append(paragraph.text)
        
        return "\n".join(text_content)
    
    def _extract_email_text(self, file_path: str) -> str:
        """Extract text from email file"""
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            email_content = f.read()
        
        # Parse email
        parser = Parser()
        email_msg = parser.parsestr(email_content)
        
        text_parts = []
        
        # Extract headers
        for header in ['From', 'To', 'Subject', 'Date']:
            if email_msg.get(header):
                text_parts.append(f"{header}: {email_msg.get(header)}")
        
        # Extract body
        if email_msg.is_multipart():
            for part in email_msg.walk():
                if part.get_content_type() == "text/plain":
                    text_parts.append(part.get_payload(decode=True).decode('utf-8', errors='ignore'))
        else:
            text_parts.append(email_msg.get_payload(decode=True).decode('utf-8', errors='ignore'))
        
        return "\n\n".join(text_parts)
    
    def _extract_text_file(self, file_path: str) -> str:
        """Extract text from plain text file"""
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read()
    
    def _create_chunks(self, text: str, source: str) -> List[DocumentChunk]:
        """Create chunks from text content"""
        chunks = []
        
        # Split text into sentences for better chunking
        sentences = self._split_into_sentences(text)
        
        current_chunk = ""
        current_tokens = 0
        chunk_id = 0
        
        for sentence in sentences:
            sentence_tokens = len(self.encoding.encode(sentence))
            
            # If adding this sentence would exceed max tokens, save current chunk
            if current_tokens + sentence_tokens > self.max_chunk_size and current_chunk:
                chunk = DocumentChunk(
                    content=current_chunk.strip(),
                    source=source,
                    chunk_id=f"{source}_{chunk_id}",
                    metadata={
                        "token_count": current_tokens,
                        "chunk_index": chunk_id
                    }
                )
                chunks.append(chunk)
                
                # Start new chunk with overlap
                overlap_text = self._get_overlap_text(current_chunk, self.chunk_overlap)
                current_chunk = overlap_text + " " + sentence
                current_tokens = len(self.encoding.encode(current_chunk))
                chunk_id += 1
            else:
                current_chunk += " " + sentence
                current_tokens += sentence_tokens
        
        # Add final chunk if there's remaining content
        if current_chunk.strip():
            chunk = DocumentChunk(
                content=current_chunk.strip(),
                source=source,
                chunk_id=f"{source}_{chunk_id}",
                metadata={
                    "token_count": current_tokens,
                    "chunk_index": chunk_id
                }
            )
            chunks.append(chunk)
        
        return chunks
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        import re
        
        # Simple sentence splitting - can be improved with more sophisticated methods
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _get_overlap_text(self, text: str, overlap_tokens: int) -> str:
        """Get overlap text for chunk continuity"""
        tokens = self.encoding.encode(text)
        if len(tokens) <= overlap_tokens:
            return text
        
        overlap_tokens_list = tokens[-overlap_tokens:]
        return self.encoding.decode(overlap_tokens_list)


class DocumentCache:
    """Simple in-memory cache for processed documents"""
    
    def __init__(self):
        self._cache: Dict[str, List[DocumentChunk]] = {}
    
    def get(self, url: str) -> Optional[List[DocumentChunk]]:
        """Get cached document chunks"""
        return self._cache.get(url)
    
    def set(self, url: str, chunks: List[DocumentChunk]) -> None:
        """Cache document chunks"""
        self._cache[url] = chunks
    
    def clear(self) -> None:
        """Clear all cached documents"""
        self._cache.clear()
    
    def size(self) -> int:
        """Get cache size"""
        return len(self._cache)


# Global instances
document_processor = DocumentProcessor()
document_cache = DocumentCache()
