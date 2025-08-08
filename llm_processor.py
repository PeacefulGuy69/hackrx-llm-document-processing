import time
import logging
from typing import List, Union, Dict, Any
import asyncio

from models import (
    QueryRequest, QueryResponse, ProcessingResult, 
    StructuredQuery, DocumentChunk, RetrievedClause, StructuredDecisionResponse
)
from document_processor import document_processor, document_cache
from query_parser import query_parser
from vector_store import semantic_searcher
from decision_engine import decision_engine
from config import config

# Set up logging
logger = logging.getLogger(__name__)


class LLMDocumentProcessor:
    """Main orchestrator for LLM-based document processing"""
    
    def __init__(self):
        self.document_processor = document_processor
        self.query_parser = query_parser
        self.semantic_searcher = semantic_searcher
        self.decision_engine = decision_engine
        self.document_cache = document_cache
    
    async def process_query_request(self, request: QueryRequest) -> QueryResponse:
        """Process the main query request from the API"""
        try:
            start_time = time.time()
            
            # Process documents if needed
            await self._ensure_documents_processed(request.documents)
            
            # Process each question
            answers = []
            structured_responses = []
            for question in request.questions:
                try:
                    answer, structured_response = await self._process_single_question(question)
                    answers.append(answer)
                    structured_responses.append(structured_response)
                except Exception as e:
                    logger.error(f"Error processing question '{question}': {str(e)}")
                    answers.append(f"Error processing question: {str(e)}")
                    structured_responses.append({
                        "decision": "error",
                        "justification": f"Error processing question: {str(e)}",
                        "confidence": 0.0
                    })
            
            processing_time = time.time() - start_time
            logger.info(f"Processed {len(request.questions)} questions in {processing_time:.2f}s")
            
            return QueryResponse(answers=answers, structured_responses=structured_responses)
            
        except Exception as e:
            logger.error(f"Error in process_query_request: {str(e)}")
            raise
    
    async def _ensure_documents_processed(self, documents: Union[str, List[str]]) -> None:
        """Ensure all documents are processed and indexed"""
        try:
            # Convert to list if single document
            if isinstance(documents, str):
                document_urls = [documents]
            else:
                document_urls = documents
            
            # Process each document
            all_chunks = []
            for url in document_urls:
                # Check cache first
                cached_chunks = self.document_cache.get(url)
                if cached_chunks:
                    logger.info(f"Using cached chunks for {url}")
                    all_chunks.extend(cached_chunks)
                else:
                    # Process document
                    logger.info(f"Processing document: {url}")
                    chunks = await self.document_processor.process_document_from_url(url)
                    
                    # Cache the chunks
                    self.document_cache.set(url, chunks)
                    all_chunks.extend(chunks)
            
            # Add all chunks to vector store
            if all_chunks:
                await self.semantic_searcher.add_documents(all_chunks)
                logger.info(f"Added {len(all_chunks)} chunks to vector store")
            
        except Exception as e:
            logger.error(f"Error processing documents: {str(e)}")
            raise
    
    async def _process_single_question(self, question: str) -> tuple[str, dict]:
        """Process a single question and return both text answer and structured response"""
        try:
            start_time = time.time()
            
            # Parse the query
            structured_query = await self.query_parser.parse_query(question)
            
            # Search for relevant clauses
            retrieved_clauses = await self.semantic_searcher.search_clauses(
                question, 
                top_k=config.TOP_K_RESULTS
            )
            
            # Make decision
            processing_result = await self.decision_engine.process_decision(
                structured_query, 
                retrieved_clauses
            )
            
            # Set processing time
            processing_result.processing_time = time.time() - start_time
            
            # Format answer and structured response
            answer = self._format_answer(processing_result)
            structured_response = self._format_structured_response(processing_result)
            
            logger.info(f"Processed question in {processing_result.processing_time:.2f}s: {question[:50]}...")
            return answer, structured_response
            
        except Exception as e:
            logger.error(f"Error processing question '{question}': {str(e)}")
            error_response = {
                "decision": "error",
                "justification": f"I apologize, but I encountered an error while processing this question: {str(e)}",
                "confidence": 0.0,
                "amount": None,
                "referenced_clauses": [],
                "processing_metadata": {"error": str(e)}
            }
            return f"I apologize, but I encountered an error while processing this question: {str(e)}", error_response
    
    def _format_answer(self, result: ProcessingResult) -> str:
        """Format the processing result into a natural language answer"""
        try:
            # Extract key components
            decision = result.decision
            reasoning = result.justification.reasoning
            confidence = result.justification.confidence_score
            amount = result.amount
            
            # Start building the answer
            answer_parts = []
            
            # Main decision
            if decision.lower() == 'approved':
                if amount:
                    answer_parts.append(f"Yes, this is covered with a benefit amount of ₹{amount:,.0f}.")
                else:
                    answer_parts.append("Yes, this is covered under the policy.")
            elif decision.lower() == 'rejected':
                answer_parts.append("No, this is not covered under the policy.")
            else:
                answer_parts.append("The coverage status requires additional information to determine.")
            
            # Add reasoning
            if reasoning and reasoning != "Unable to determine from available information":
                # Clean up the reasoning text
                clean_reasoning = self._clean_reasoning_text(reasoning)
                answer_parts.append(clean_reasoning)
            
            # Add specific details from structured query if relevant
            structured_query = result.structured_query
            details = []
            
            if structured_query.procedure:
                details.append(f"procedure: {structured_query.procedure}")
            if structured_query.policy_duration:
                details.append(f"policy duration: {structured_query.policy_duration}")
            if structured_query.age:
                details.append(f"age: {structured_query.age}")
            
            if details and len(details) <= 3:  # Only add if not too many details
                details_str = ", ".join(details)
                answer_parts.append(f"Based on the query parameters ({details_str}).")
            
            # Add confidence if low
            if confidence < 0.6:
                answer_parts.append("Please note that this assessment has moderate confidence and may require manual review.")
            
            # Join all parts
            final_answer = " ".join(answer_parts)
            
            # Ensure reasonable length
            if len(final_answer) > 500:
                # Truncate but keep essential information
                essential_parts = answer_parts[:2]  # Keep decision and main reasoning
                final_answer = " ".join(essential_parts)
            
            return final_answer
            
        except Exception as e:
            logger.error(f"Error formatting answer: {str(e)}")
            return "The query has been processed, but there was an error formatting the response."
    
    def _clean_reasoning_text(self, reasoning: str) -> str:
        """Clean and format reasoning text"""
        # Remove JSON-like formatting
        import re
        
        # Remove JSON structure words
        reasoning = re.sub(r'\b(decision|reasoning|confidence|amount)\b\s*[:=]\s*', '', reasoning, flags=re.IGNORECASE)
        
        # Remove quotes and brackets
        reasoning = re.sub(r'["\[\]{}]', '', reasoning)
        
        # Clean up multiple spaces
        reasoning = re.sub(r'\s+', ' ', reasoning)
        
        # Capitalize first letter
        reasoning = reasoning.strip()
        if reasoning:
            reasoning = reasoning[0].upper() + reasoning[1:]
        
        # Ensure it ends with proper punctuation
        if reasoning and not reasoning.endswith(('.', '!', '?')):
            reasoning += '.'
        
        return reasoning
    
    def _format_structured_response(self, result: ProcessingResult) -> dict:
        """Format the processing result into a structured JSON response as per problem statement"""
        try:
            # Extract key components
            decision = result.decision
            reasoning = result.justification.reasoning
            confidence = result.justification.confidence_score
            amount = result.amount
            referenced_clauses = result.justification.referenced_clauses
            
            # Create referenced clauses summary
            clause_summaries = []
            for clause in referenced_clauses[:5]:  # Top 5 clauses
                summary = f"From {clause.source}: {clause.content[:100]}..."
                clause_summaries.append(summary)
            
            # Create structured response
            structured_response = {
                "decision": decision.lower(),  # approved, rejected, or pending
                "amount": amount,
                "justification": reasoning,
                "confidence": round(confidence, 3),
                "referenced_clauses": clause_summaries,
                "processing_metadata": {
                    "query_type": result.structured_query.query_type.value,
                    "processing_time": round(result.processing_time, 2),
                    "clauses_analyzed": len(referenced_clauses),
                    "extracted_entities": {
                        "age": result.structured_query.age,
                        "gender": result.structured_query.gender,
                        "procedure": result.structured_query.procedure,
                        "location": result.structured_query.location,
                        "policy_duration": result.structured_query.policy_duration
                    }
                }
            }
            
            return structured_response
            
        except Exception as e:
            logger.error(f"Error formatting structured response: {str(e)}")
            return {
                "decision": "error",
                "amount": None,
                "justification": "Error formatting response",
                "confidence": 0.0,
                "referenced_clauses": [],
                "processing_metadata": {"error": str(e)}
            }
    
    async def get_system_stats(self) -> Dict[str, Any]:
        """Get system statistics"""
        try:
            vector_stats = self.semantic_searcher.get_stats()
            cache_stats = {
                "cached_documents": self.document_cache.size(),
                "cache_sources": list(self.document_cache._cache.keys()) if hasattr(self.document_cache, '_cache') else []
            }
            
            return {
                "vector_store": vector_stats,
                "document_cache": cache_stats,
                "config": {
                    "llm_model": config.LLM_MODEL,
                    "embedding_model": config.EMBEDDING_MODEL,
                    "max_tokens": config.MAX_TOKENS,
                    "top_k_results": config.TOP_K_RESULTS
                }
            }
        except Exception as e:
            logger.error(f"Error getting system stats: {str(e)}")
            return {"error": str(e)}
    
    async def clear_cache(self) -> None:
        """Clear all cached data"""
        try:
            self.document_cache.clear()
            self.semantic_searcher.clear()
            logger.info("Cleared all cached data")
        except Exception as e:
            logger.error(f"Error clearing cache: {str(e)}")
            raise
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check"""
        try:
            # Test vector store
            vector_stats = self.semantic_searcher.get_stats()
            
            # Test LLM connection (simple test)
            test_query = "test query"
            structured_query = await self.query_parser.parse_query(test_query)
            
            return {
                "status": "healthy",
                "components": {
                    "vector_store": "healthy" if vector_stats["index_size"] >= 0 else "error",
                    "query_parser": "healthy" if structured_query else "error",
                    "document_processor": "healthy",
                    "decision_engine": "healthy"
                },
                "stats": vector_stats
            }
        except Exception as e:
            logger.error(f"Health check failed: {str(e)}")
            return {
                "status": "unhealthy",
                "error": str(e)
            }


# Global instance
llm_processor = LLMDocumentProcessor()
