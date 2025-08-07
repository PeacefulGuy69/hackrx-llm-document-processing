import logging
import json
from typing import List, Dict, Any, Optional
from openai import AsyncOpenAI

from models import StructuredQuery, RetrievedClause, DecisionJustification, ProcessingResult
from config import config

# Set up logging
logger = logging.getLogger(__name__)


class DecisionEngine:
    """Processes retrieved clauses and makes decisions based on policy logic"""
    
    def __init__(self):
        self.client = AsyncOpenAI(api_key=config.OPENAI_API_KEY)
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
            
            response = await self.client.chat.completions.create(
                model=config.LLM_MODEL,
                messages=[
                    {"role": "system", "content": self.decision_prompt},
                    {"role": "user", "content": context}
                ],
                temperature=config.TEMPERATURE,
                max_tokens=1000
            )
            
            content = response.choices[0].message.content
            
            # Try to parse JSON response
            try:
                return json.loads(content)
            except json.JSONDecodeError:
                # Parse structured text response
                return self._parse_text_decision(content)
                
        except Exception as e:
            logger.error(f"Error in LLM decision generation: {str(e)}")
            return {'decision': 'pending', 'reasoning': 'Error in decision processing'}
    
    def _build_decision_prompt(self) -> str:
        """Build the decision-making prompt for the LLM"""
        return """You are an expert insurance claims processor and policy analyst. Your job is to analyze insurance queries against policy documents and make accurate decisions.

Given a structured query and relevant policy clauses, you must:

1. Analyze the query requirements against the policy terms
2. Determine if the claim/coverage is approved, rejected, or needs more information
3. Calculate any applicable amounts based on policy terms
4. Provide clear reasoning with specific clause references
5. Assign a confidence score (0.0 to 1.0)

Decision Rules:
- APPROVED: Clear coverage exists, all conditions met
- REJECTED: Explicitly excluded or conditions not met
- PENDING: Insufficient information or ambiguous case

For insurance coverage queries:
- Check waiting periods, exclusions, and coverage limits
- Consider age, pre-existing conditions, and policy duration
- Verify if the procedure/treatment is covered

For claim processing:
- Validate claim against policy terms
- Calculate coverage amounts considering sub-limits and deductibles
- Check for any applicable waiting periods

Return your analysis as a JSON object with these fields:
{
    "decision": "approved|rejected|pending",
    "amount": number or null,
    "reasoning": "detailed explanation with clause references",
    "confidence": 0.0-1.0,
    "key_factors": ["list of key factors considered"],
    "applicable_clauses": ["specific clause references"],
    "conditions": ["any conditions that must be met"]
}

Be thorough, accurate, and always reference specific clauses in your reasoning."""
    
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
