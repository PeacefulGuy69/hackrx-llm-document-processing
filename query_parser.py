import re
import json
import logging
from typing import Dict, Any, Optional, List
import google.generativeai as genai

from models import StructuredQuery, QueryType
from config import config

# Set up logging
logger = logging.getLogger(__name__)


class QueryParser:
    """Parses natural language queries into structured format"""
    
    def __init__(self):
        genai.configure(api_key=config.GEMINI_API_KEY)
        self.model = genai.GenerativeModel(config.LLM_MODEL)
        self.extraction_prompt = self._build_extraction_prompt()
    
    async def parse_query(self, query: str) -> StructuredQuery:
        """Parse natural language query into structured format"""
        try:
            # Use LLM to extract structured information
            structured_data = await self._extract_with_llm(query)
            
            # Enhance with regex-based extraction
            enhanced_data = self._enhance_with_regex(query, structured_data)
            
            # Create structured query object
            structured_query = StructuredQuery(
                age=enhanced_data.get('age'),
                gender=enhanced_data.get('gender'),
                procedure=enhanced_data.get('procedure'),
                location=enhanced_data.get('location'),
                policy_duration=enhanced_data.get('policy_duration'),
                policy_type=enhanced_data.get('policy_type'),
                amount=enhanced_data.get('amount'),
                query_type=self._determine_query_type(query, enhanced_data),
                original_query=query,
                extracted_entities=enhanced_data
            )
            
            logger.info(f"Parsed query: {query[:50]}... -> {structured_query.dict()}")
            return structured_query
            
        except Exception as e:
            logger.error(f"Error parsing query '{query}': {str(e)}")
            # Return basic structured query on error
            return StructuredQuery(
                original_query=query,
                query_type=QueryType.GENERAL,
                extracted_entities={}
            )
    
    async def _extract_with_llm(self, query: str) -> Dict[str, Any]:
        """Extract structured information using LLM"""
        try:
            prompt = f"{self.extraction_prompt}\n\nQuery: {query}"
            
            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.1,
                    max_output_tokens=500
                )
            )
            
            content = response.text
            
            # Try to parse JSON response
            try:
                return json.loads(content)
            except json.JSONDecodeError:
                # If JSON parsing fails, try to extract from text
                return self._parse_llm_text_response(content)
                
        except Exception as e:
            logger.error(f"Error in LLM extraction: {str(e)}")
            return {}
    
    def _build_extraction_prompt(self) -> str:
        """Build prompt for structured information extraction"""
        return """You are an expert at extracting structured information from insurance and medical queries.

Extract the following information from the user query and return it as a JSON object:

- age: Age of the person (integer)
- gender: Gender (M/F/Male/Female or null)
- procedure: Medical procedure or treatment mentioned
- location: City, state, or location mentioned
- policy_duration: Duration or age of policy (e.g., "3 months", "1 year")
- policy_type: Type of insurance policy mentioned
- amount: Any monetary amount mentioned (as number)
- medical_condition: Any medical condition mentioned
- urgency: Urgency level (urgent/normal/routine)
- coverage_type: Type of coverage being asked about
- time_period: Any time period mentioned
- hospital: Hospital or medical facility mentioned

Return ONLY a valid JSON object with the extracted information. Use null for missing information.

Example:
Query: "46-year-old male, knee surgery in Pune, 3-month-old insurance policy"
Response: {
  "age": 46,
  "gender": "male",
  "procedure": "knee surgery",
  "location": "Pune",
  "policy_duration": "3 months",
  "policy_type": "insurance policy",
  "amount": null,
  "medical_condition": "knee problem",
  "urgency": "normal",
  "coverage_type": "surgical",
  "time_period": null,
  "hospital": null
}"""
    
    def _parse_llm_text_response(self, content: str) -> Dict[str, Any]:
        """Parse LLM text response when JSON parsing fails"""
        extracted = {}
        
        # Try to find JSON-like patterns in the text
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass
        
        # Fallback to regex extraction
        patterns = {
            'age': r'(?:age|years?)\D*(\d+)',
            'gender': r'(male|female|m|f)\b',
            'amount': r'(?:rs|rupees?|₹)\s*(\d+(?:,\d+)*)',
            'location': r'(?:in|at|from)\s+([A-Za-z\s]+?)(?:\s|,|$)'
        }
        
        for key, pattern in patterns.items():
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                extracted[key] = match.group(1).strip()
        
        return extracted
    
    def _enhance_with_regex(self, query: str, llm_data: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance LLM extraction with regex-based patterns"""
        enhanced = llm_data.copy()
        
        # Age extraction
        if not enhanced.get('age'):
            age_match = re.search(r'(\d+)[-\s]*(?:year|yr|y)[-\s]*old|(\d+)[-\s]*(?:M|F)', query, re.IGNORECASE)
            if age_match:
                enhanced['age'] = int(age_match.group(1) or age_match.group(2))
        
        # Gender extraction
        if not enhanced.get('gender'):
            gender_match = re.search(r'\b(\d+)\s*(M|F|male|female)\b', query, re.IGNORECASE)
            if gender_match:
                enhanced['gender'] = gender_match.group(2).lower()
        
        # Policy duration extraction
        if not enhanced.get('policy_duration'):
            duration_match = re.search(r'(\d+)[-\s]*(?:month|year|day)[-\s]*(?:old|policy)', query, re.IGNORECASE)
            if duration_match:
                enhanced['policy_duration'] = duration_match.group(0)
        
        # Amount extraction
        if not enhanced.get('amount'):
            amount_patterns = [
                r'(?:rs|rupees?|₹)\s*(\d+(?:,\d+)*)',
                r'(\d+(?:,\d+)*)\s*(?:rs|rupees?|₹)',
                r'(\d+(?:,\d+)*)\s*(?:lakh|crore)'
            ]
            for pattern in amount_patterns:
                amount_match = re.search(pattern, query, re.IGNORECASE)
                if amount_match:
                    amount_str = amount_match.group(1).replace(',', '')
                    enhanced['amount'] = float(amount_str)
                    break
        
        # Common medical procedures
        procedures = [
            'surgery', 'operation', 'treatment', 'therapy', 'consultation',
            'knee surgery', 'heart surgery', 'cataract', 'bypass', 'transplant',
            'chemotherapy', 'dialysis', 'physiotherapy', 'dental', 'maternity'
        ]
        
        if not enhanced.get('procedure'):
            for procedure in procedures:
                if procedure.lower() in query.lower():
                    enhanced['procedure'] = procedure
                    break
        
        # Indian cities for location extraction
        indian_cities = [
            'mumbai', 'delhi', 'bangalore', 'hyderabad', 'ahmedabad', 'chennai',
            'kolkata', 'surat', 'pune', 'jaipur', 'lucknow', 'kanpur', 'nagpur',
            'visakhapatnam', 'indore', 'thane', 'bhopal', 'patna', 'vadodara',
            'ghaziabad', 'ludhiana', 'agra', 'nashik', 'faridabad', 'meerut'
        ]
        
        if not enhanced.get('location'):
            for city in indian_cities:
                if city.lower() in query.lower():
                    enhanced['location'] = city.title()
                    break
        
        return enhanced
    
    def _determine_query_type(self, query: str, extracted_data: Dict[str, Any]) -> QueryType:
        """Determine the type of query based on content"""
        query_lower = query.lower()
        
        # Coverage-related keywords
        coverage_keywords = ['cover', 'coverage', 'covered', 'include', 'included', 'eligible']
        if any(keyword in query_lower for keyword in coverage_keywords):
            return QueryType.COVERAGE
        
        # Claim-related keywords
        claim_keywords = ['claim', 'reimbursement', 'reimburse', 'payout', 'settlement']
        if any(keyword in query_lower for keyword in claim_keywords):
            return QueryType.CLAIM
        
        # Policy-related keywords
        policy_keywords = ['policy', 'premium', 'renewal', 'terms', 'conditions']
        if any(keyword in query_lower for keyword in policy_keywords):
            return QueryType.POLICY
        
        # If medical procedure is mentioned, likely coverage query
        if extracted_data.get('procedure'):
            return QueryType.COVERAGE
        
        return QueryType.GENERAL


# Global instance
query_parser = QueryParser()
