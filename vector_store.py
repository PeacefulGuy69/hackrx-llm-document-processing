import os
import pickle
import logging
import numpy as np
from typing import List, Optional, Dict, Any
import faiss
from sentence_transformers import SentenceTransformer
import google.generativeai as genai

from models import DocumentChunk, RetrievedClause
from config import config

# Set up logging
logger = logging.getLogger(__name__)


class VectorStore:
    """FAISS-based vector store for semantic search"""
    
    def __init__(self):
        self.dimension = config.VECTOR_DIMENSION
        self.index = None
        self.chunks: List[DocumentChunk] = []
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')  # Fast and efficient
        genai.configure(api_key=config.GEMINI_API_KEY)
        self.index_path = config.FAISS_INDEX_PATH
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        
        # Initialize FAISS index
        self._initialize_index()
    
    def _initialize_index(self):
        """Initialize FAISS index"""
        try:
            # Get actual embedding dimension from the model
            test_embedding = self.embedding_model.encode(["test"])
            actual_dimension = test_embedding.shape[1]
            
            # Update dimension to match the actual model
            self.dimension = actual_dimension
            
            # Try to load existing index
            if os.path.exists(f"{self.index_path}.index"):
                try:
                    self.index = faiss.read_index(f"{self.index_path}.index")
                    
                    # Check if dimensions match
                    if self.index.d != self.dimension:
                        logger.warning(f"Existing index dimension ({self.index.d}) doesn't match model dimension ({self.dimension}). Creating new index.")
                        self.index = faiss.IndexFlatIP(self.dimension)
                        self.chunks = []
                    else:
                        # Load chunks metadata
                        if os.path.exists(f"{self.index_path}.chunks"):
                            with open(f"{self.index_path}.chunks", 'rb') as f:
                                self.chunks = pickle.load(f)
                        
                        logger.info(f"Loaded existing FAISS index with {len(self.chunks)} chunks")
                except Exception as e:
                    logger.warning(f"Error loading existing index: {e}. Creating new index.")
                    self.index = faiss.IndexFlatIP(self.dimension)
                    self.chunks = []
            else:
                # Create new index
                self.index = faiss.IndexFlatIP(self.dimension)  # Inner Product for cosine similarity
                logger.info(f"Created new FAISS index with dimension {self.dimension}")
                
        except Exception as e:
            logger.error(f"Error initializing FAISS index: {str(e)}")
            self.index = faiss.IndexFlatIP(self.dimension)
    
    async def add_documents(self, chunks: List[DocumentChunk]) -> None:
        """Add document chunks to the vector store"""
        try:
            if not chunks:
                return
            
            # Generate embeddings for chunks
            texts = [chunk.content for chunk in chunks]
            embeddings = await self._generate_embeddings(texts)
            
            # Ensure embeddings have correct dimension
            if embeddings.shape[1] != self.dimension:
                logger.error(f"Embedding dimension mismatch: got {embeddings.shape[1]}, expected {self.dimension}")
                # Recreate index with correct dimension
                self.dimension = embeddings.shape[1]
                self.index = faiss.IndexFlatIP(self.dimension)
                self.chunks = []
                logger.info(f"Recreated FAISS index with dimension {self.dimension}")
            
            # Normalize embeddings for cosine similarity
            embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
            
            # Add embeddings to chunks
            for chunk, embedding in zip(chunks, embeddings):
                chunk.embedding = embedding.tolist()
            
            # Add to FAISS index
            self.index.add(embeddings.astype('float32'))
            
            # Store chunks
            self.chunks.extend(chunks)
            
            # Save index and chunks
            await self._save_index()
            
            logger.info(f"Added {len(chunks)} chunks to vector store")
            
        except Exception as e:
            logger.error(f"Error adding documents to vector store: {str(e)}")
            raise
    
    async def search(self, query: str, top_k: int = None) -> List[RetrievedClause]:
        """Search for similar chunks using semantic similarity"""
        try:
            if top_k is None:
                top_k = config.TOP_K_RESULTS
            
            if not self.chunks or self.index.ntotal == 0:
                logger.warning("No documents in vector store")
                return []
            
            # Generate query embedding
            query_embedding = await self._generate_embeddings([query])
            query_embedding = query_embedding / np.linalg.norm(query_embedding, axis=1, keepdims=True)
            
            # Search in FAISS index
            scores, indices = self.index.search(query_embedding.astype('float32'), min(top_k, len(self.chunks)))
            
            # Create retrieved clauses
            retrieved_clauses = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < len(self.chunks):
                    chunk = self.chunks[idx]
                    clause = RetrievedClause(
                        content=chunk.content,
                        source=chunk.source,
                        page_number=chunk.page_number,
                        similarity_score=float(score),
                        metadata=chunk.metadata
                    )
                    retrieved_clauses.append(clause)
            
            # Sort by similarity score (descending)
            retrieved_clauses.sort(key=lambda x: x.similarity_score, reverse=True)
            
            logger.info(f"Retrieved {len(retrieved_clauses)} clauses for query: {query[:50]}...")
            return retrieved_clauses
            
        except Exception as e:
            logger.error(f"Error searching vector store: {str(e)}")
            return []
    
    async def _generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for texts"""
        try:
            # Use Gemini embeddings for better quality
            embeddings = []
            for text in texts:
                result = genai.embed_content(
                    model=config.EMBEDDING_MODEL,
                    content=text
                )
                embeddings.append(result['embedding'])
            
            return np.array(embeddings)
            
        except Exception as e:
            logger.warning(f"Gemini embedding failed, using local model: {str(e)}")
            # Fallback to local model
            embeddings = self.embedding_model.encode(texts)
            
            # Pad or truncate to match dimension
            if embeddings.shape[1] != self.dimension:
                if embeddings.shape[1] < self.dimension:
                    # Pad with zeros
                    padding = np.zeros((embeddings.shape[0], self.dimension - embeddings.shape[1]))
                    embeddings = np.concatenate([embeddings, padding], axis=1)
                else:
                    # Truncate
                    embeddings = embeddings[:, :self.dimension]
            
            return embeddings
    
    async def _save_index(self):
        """Save FAISS index and chunks to disk"""
        try:
            # Save FAISS index
            faiss.write_index(self.index, f"{self.index_path}.index")
            
            # Save chunks metadata
            with open(f"{self.index_path}.chunks", 'wb') as f:
                pickle.dump(self.chunks, f)
            
            logger.info("Saved FAISS index and chunks")
            
        except Exception as e:
            logger.error(f"Error saving index: {str(e)}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get vector store statistics"""
        return {
            "total_chunks": len(self.chunks),
            "index_size": self.index.ntotal if self.index else 0,
            "dimension": self.dimension,
            "sources": list(set(chunk.source for chunk in self.chunks))
        }
    
    def clear(self):
        """Clear all data from vector store"""
        try:
            self.index = faiss.IndexFlatIP(self.dimension)
            self.chunks = []
            
            # Remove saved files
            for file_path in [f"{self.index_path}.index", f"{self.index_path}.chunks"]:
                if os.path.exists(file_path):
                    os.remove(file_path)
            
            logger.info("Cleared vector store")
            
        except Exception as e:
            logger.error(f"Error clearing vector store: {str(e)}")


class SemanticSearcher:
    """High-level semantic search interface"""
    
    def __init__(self):
        self.vector_store = VectorStore()
    
    async def add_documents(self, chunks: List[DocumentChunk]) -> None:
        """Add documents to the search index"""
        await self.vector_store.add_documents(chunks)
    
    async def search_clauses(self, query: str, top_k: int = None) -> List[RetrievedClause]:
        """Search for relevant clauses"""
        return await self.vector_store.search(query, top_k)
    
    async def search_with_filters(self, query: str, source_filter: Optional[str] = None, 
                                min_score: float = 0.0, top_k: int = None) -> List[RetrievedClause]:
        """Search with additional filters"""
        clauses = await self.search_clauses(query, top_k)
        
        # Apply filters
        filtered_clauses = []
        for clause in clauses:
            # Source filter
            if source_filter and source_filter not in clause.source:
                continue
            
            # Minimum score filter
            if clause.similarity_score < min_score:
                continue
            
            filtered_clauses.append(clause)
        
        return filtered_clauses
    
    def get_stats(self) -> Dict[str, Any]:
        """Get search system statistics"""
        return self.vector_store.get_stats()
    
    def clear(self):
        """Clear all indexed data"""
        self.vector_store.clear()


# Global instance
semantic_searcher = SemanticSearcher()
