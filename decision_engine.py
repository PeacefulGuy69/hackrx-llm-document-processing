import logging
import json
from typing import List, Dict, Any, Optional
import google.generativeai as genai

from models import StructuredQuery, RetrievedClause, DecisionJustification, ProcessingResult
from config import config

# Set up logging
logger = logging.getLogger(__name__)


class DecisionEngine:
    """Processes retrieved clauses and makes decisions based on policy logic"""
    
    def __init__(self):
        genai.configure(api_key=config.GEMINI_API_KEY)
        self.model = genai.GenerativeModel(config.LLM_MODEL)
        self.decision_prompt = self._build_decision_prompt()
    
    async def process_decision(self, structured_query: StructuredQuery, 
                             retrieved_clauses: List[RetrievedClause]) -> ProcessingResult:
        """Process decision based on query and retrieved clauses"""
        try:
            # Generate decision using LLM
            decision_data = await self._generate_decision(structured_query, retrieved_clauses)
            
            # Extract decision components
            decision = decision_data.get('decision', 'pending')
            amount = decision_data.get('amount')
            reasoning = decision_data.get('reasoning', 'Unable to determine from available information')
            confidence = decision_data.get('confidence', 0.5)
            
            # Create decision justification
            justification = DecisionJustification(
                decision=decision,
                reasoning=reasoning,
                referenced_clauses=retrieved_clauses[:5],  # Top 5 most relevant
                confidence_score=confidence
            )
            
            # Create processing result
            result = ProcessingResult(
                decision=decision,
                amount=amount,
                justification=justification,
                structured_query=structured_query,
                processing_time=0.0,  # Will be set by the main processor
                metadata={
                    'clauses_processed': len(retrieved_clauses),
                    'decision_method': 'llm_analysis',
                    'confidence_level': self._get_confidence_level(confidence)
                }
            )
            
            logger.info(f"Decision processed: {decision} with confidence {confidence}")
            return result
            
        except Exception as e:
            logger.error(f"Error processing decision: {str(e)}")
            
            # Return fallback decision
            return self._create_fallback_result(structured_query, retrieved_clauses)
    
    async def _generate_decision(self, structured_query: StructuredQuery, 
                               retrieved_clauses: List[RetrievedClause]) -> Dict[str, Any]:
        """Generate decision using LLM analysis"""
        try:
            # Prepare context for LLM
            context = self._prepare_decision_context(structured_query, retrieved_clauses)
            
            prompt = f"{self.decision_prompt}\n\n{context}"
            
            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=config.TEMPERATURE,
                    max_output_tokens=1000
                )
            )
            
            # Check if response was blocked by safety filters
            if response.candidates and len(response.candidates) > 0:
                candidate = response.candidates[0]
                if candidate.finish_reason and candidate.finish_reason != 1:  # 1 = STOP (successful)
                    logger.warning(f"Response blocked by safety filters. Finish reason: {candidate.finish_reason}")
                    # Use rule-based fallback when content is blocked
                    return self._generate_rule_based_decision(structured_query, retrieved_clauses)
            
            # Check if we have valid content
            if not hasattr(response, 'text') or not response.text:
                logger.warning("No text content in response, using rule-based fallback")
                return self._generate_rule_based_decision(structured_query, retrieved_clauses)
            
            content = response.text
            
            # Try to parse JSON response
            try:
                return json.loads(content)
            except json.JSONDecodeError:
                # Parse structured text response
                return self._parse_text_decision(content)
                
        except Exception as e:
            logger.error(f"Error in LLM decision generation: {str(e)}")
            # Use rule-based fallback instead of just returning error
            return self._generate_rule_based_decision(structured_query, retrieved_clauses)
    
    def _build_decision_prompt(self) -> str:
        """Build the decision-making prompt for the LLM"""
        return """You are a professional insurance policy analyst. Analyze insurance coverage questions against policy documents to provide accurate assessments.

Your task is to:
1. Review the customer query against policy terms and conditions
2. Determine coverage status: approved, rejected, or pending (needs more information)
3. Calculate applicable amounts based on policy benefits
4. Provide clear explanations with policy clause references
5. Assign a confidence score from 0.0 to 1.0

Coverage Assessment Guidelines:
- APPROVED: Coverage is clearly provided, conditions are satisfied
- REJECTED: Coverage is explicitly excluded or conditions are not met  
- PENDING: Insufficient information or requires clarification

For coverage analysis:
- Review waiting periods, benefit limits, and eligibility criteria
- Consider policy duration, age restrictions, and covered services
- Verify if the requested service or treatment is included

For benefit calculations:
- Apply policy limits, sub-limits, and deductibles as specified
- Check for applicable waiting periods or restrictions
- Calculate coverage amounts based on policy schedule

Please respond with a JSON object containing:
{
    "decision": "approved|rejected|pending",
    "amount": number or null,
    "reasoning": "detailed explanation with policy references",
    "confidence": 0.0-1.0,
    "key_factors": ["important factors considered"],
    "applicable_clauses": ["relevant policy sections"],
    "conditions": ["requirements that must be satisfied"]
}

Provide thorough, accurate analysis with specific policy clause references."""
    
    def _prepare_decision_context(self, structured_query: StructuredQuery, 
                                retrieved_clauses: List[RetrievedClause]) -> str:
        """Prepare context for LLM decision making"""
        context_parts = []
        
        # Add query information
        context_parts.append("=== QUERY ANALYSIS ===")
        context_parts.append(f"Original Query: {structured_query.original_query}")
        context_parts.append(f"Query Type: {structured_query.query_type}")
        
        if structured_query.age:
            context_parts.append(f"Age: {structured_query.age}")
        if structured_query.gender:
            context_parts.append(f"Gender: {structured_query.gender}")
        if structured_query.procedure:
            context_parts.append(f"Procedure: {structured_query.procedure}")
        if structured_query.location:
            context_parts.append(f"Location: {structured_query.location}")
        if structured_query.policy_duration:
            context_parts.append(f"Policy Duration: {structured_query.policy_duration}")
        if structured_query.amount:
            context_parts.append(f"Amount: {structured_query.amount}")
        
        # Add relevant clauses
        context_parts.append("\n=== RELEVANT POLICY CLAUSES ===")
        for i, clause in enumerate(retrieved_clauses[:10], 1):  # Top 10 clauses
            context_parts.append(f"\nClause {i} (Similarity: {clause.similarity_score:.3f}):")
            context_parts.append(f"Source: {clause.source}")
            context_parts.append(f"Content: {clause.content}")
            if clause.metadata:
                context_parts.append(f"Metadata: {clause.metadata}")
        
        context_parts.append("\n=== DECISION REQUIRED ===")
        context_parts.append("Analyze the above information and provide a structured decision.")
        
        return "\n".join(context_parts)
    
    def _parse_text_decision(self, content: str) -> Dict[str, Any]:
        """Parse text-based decision response"""
        decision_data = {
            'decision': 'pending',
            'reasoning': content,
            'confidence': 0.5
        }
        
        # Extract decision
        content_lower = content.lower()
        if 'approved' in content_lower or 'covered' in content_lower:
            decision_data['decision'] = 'approved'
            decision_data['confidence'] = 0.8
        elif 'rejected' in content_lower or 'not covered' in content_lower or 'excluded' in content_lower:
            decision_data['decision'] = 'rejected'
            decision_data['confidence'] = 0.8
        
        # Extract amount
        import re
        amount_pattern = r'(?:rs|rupees?|₹)\s*(\d+(?:,\d+)*)'
        amount_match = re.search(amount_pattern, content, re.IGNORECASE)
        if amount_match:
            decision_data['amount'] = float(amount_match.group(1).replace(',', ''))
        
        return decision_data
    
    def _generate_rule_based_decision(self, structured_query: StructuredQuery, 
                                    retrieved_clauses: List[RetrievedClause]) -> Dict[str, Any]:
        """Generate decision using rule-based analysis when LLM fails"""
        try:
            # Analyze query content and clauses using rules
            decision_data = {
                'decision': 'pending',
                'reasoning': 'Analysis based on policy clauses and rule-based processing',
                'confidence': 0.6,
                'key_factors': [],
                'applicable_clauses': [],
                'conditions': []
            }
            
            if not retrieved_clauses:
                decision_data.update({
                    'decision': 'pending',
                    'reasoning': 'No relevant policy clauses found for this query',
                    'confidence': 0.3
                })
                return decision_data
            
            # Analyze top clauses for decision indicators
            high_relevance_clauses = [c for c in retrieved_clauses if c.similarity_score > 0.7]
            
            # Check for coverage indicators
            coverage_keywords = ['covered', 'includes', 'benefit', 'eligible', 'reimbursement']
            exclusion_keywords = ['excluded', 'not covered', 'waiting period', 'pre-existing', 'limitation']
            
            coverage_score = 0
            exclusion_score = 0
            
            for clause in high_relevance_clauses:
                content_lower = clause.content.lower()
                
                # Count coverage indicators
                coverage_score += sum(1 for keyword in coverage_keywords if keyword in content_lower)
                
                # Count exclusion indicators
                exclusion_score += sum(1 for keyword in exclusion_keywords if keyword in content_lower)
                
                decision_data['applicable_clauses'].append(f"Clause from {clause.source}")
            
            # Make decision based on scores
            if coverage_score > exclusion_score and coverage_score > 0:
                decision_data.update({
                    'decision': 'approved',
                    'reasoning': f'Found {coverage_score} coverage indicators in relevant policy clauses',
                    'confidence': min(0.8, 0.5 + (coverage_score * 0.1))
                })
            elif exclusion_score > coverage_score and exclusion_score > 0:
                decision_data.update({
                    'decision': 'rejected',
                    'reasoning': f'Found {exclusion_score} exclusion indicators in relevant policy clauses',
                    'confidence': min(0.8, 0.5 + (exclusion_score * 0.1))
                })
            else:
                decision_data.update({
                    'decision': 'pending',
                    'reasoning': 'Ambiguous coverage status requires additional information to determine',
                    'confidence': 0.4
                })
            
            return decision_data
            
        except Exception as e:
            logger.error(f"Error in rule-based decision generation: {str(e)}")
            return {
                'decision': 'pending',
                'reasoning': 'Error in decision processing. Please note that this assessment has moderate confidence and may require manual review',
                'confidence': 0.3
            }
    
    def _create_fallback_result(self, structured_query: StructuredQuery, 
                              retrieved_clauses: List[RetrievedClause]) -> ProcessingResult:
        """Create fallback result when LLM processing fails"""
        # Simple rule-based fallback
        decision = self._simple_rule_based_decision(structured_query, retrieved_clauses)
        
        justification = DecisionJustification(
            decision=decision,
            reasoning="Decision made using fallback rule-based system due to processing error",
            referenced_clauses=retrieved_clauses[:3],
            confidence_score=0.3
        )
        
        return ProcessingResult(
            decision=decision,
            amount=None,
            justification=justification,
            structured_query=structured_query,
            processing_time=0.0,
            metadata={'decision_method': 'fallback_rules'}
        )
    
    def _simple_rule_based_decision(self, structured_query: StructuredQuery, 
                                  retrieved_clauses: List[RetrievedClause]) -> str:
        """Simple rule-based decision for fallback"""
        # Basic rules based on common insurance patterns
        
        # If no relevant clauses found
        if not retrieved_clauses:
            return "pending"
        
        # Check for high similarity matches
        high_similarity_clauses = [c for c in retrieved_clauses if c.similarity_score > 0.8]
        if high_similarity_clauses:
            # Look for exclusion keywords in high similarity clauses
            exclusion_keywords = ['excluded', 'not covered', 'waiting period', 'pre-existing']
            for clause in high_similarity_clauses:
                content_lower = clause.content.lower()
                if any(keyword in content_lower for keyword in exclusion_keywords):
                    return "rejected"
            
            # If high similarity and no exclusions, likely approved
            return "approved"
        
        # Default to pending for ambiguous cases
        return "pending"
    
    def _get_confidence_level(self, confidence: float) -> str:
        """Convert confidence score to level"""
        if confidence >= 0.8:
            return "high"
        elif confidence >= 0.6:
            return "medium"
        else:
            return "low"


# Global instance
decision_engine = DecisionEngine()
