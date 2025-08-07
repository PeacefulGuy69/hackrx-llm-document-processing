import asyncio
import httpx
import json
from typing import Dict, Any


class HackRxAPITester:
    """Test client for the HackRx API"""
    
    def __init__(self, base_url: str = "http://localhost:8000", 
                 token: str = "60359a637b23864b320999e8d98517f239970ee339c266bde110414ce8fb9ed1"):
        self.base_url = base_url
        self.headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }
    
    async def test_health(self) -> Dict[str, Any]:
        """Test health endpoint"""
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{self.base_url}/health")
            return response.json()
    
    async def test_sample_query(self) -> Dict[str, Any]:
        """Test with sample query from the problem statement"""
        sample_request = {
            "documents": "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D",
            "questions": [
                "What is the grace period for premium payment under the National Parivar Mediclaim Plus Policy?",
                "What is the waiting period for pre-existing diseases (PED) to be covered?",
                "Does this policy cover maternity expenses, and what are the conditions?",
                "What is the waiting period for cataract surgery?",
                "Are the medical expenses for an organ donor covered under this policy?"
            ]
        }
        
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                f"{self.base_url}/api/v1/hackrx/run",
                headers=self.headers,
                json=sample_request
            )
            return response.json()
    
    async def test_custom_query(self, documents: str, questions: list) -> Dict[str, Any]:
        """Test with custom query"""
        request_data = {
            "documents": documents,
            "questions": questions
        }
        
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                f"{self.base_url}/api/v1/hackrx/run",
                headers=self.headers,
                json=request_data
            )
            return response.json()
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get system statistics"""
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{self.base_url}/api/v1/stats",
                headers=self.headers
            )
            return response.json()


async def main():
    """Main test function"""
    tester = HackRxAPITester()
    
    print("🔧 Testing LLM Document Processing System")
    print("=" * 50)
    
    # Test health
    print("\n1. Testing Health Endpoint...")
    try:
        health_result = await tester.test_health()
        print(f"✅ Health Status: {health_result.get('status', 'unknown')}")
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return
    
    # Test sample query
    print("\n2. Testing Sample Query...")
    try:
        result = await tester.test_sample_query()
        print(f"✅ Processed {len(result.get('answers', []))} questions")
        
        # Print first few answers
        answers = result.get('answers', [])
        for i, answer in enumerate(answers[:3], 1):
            print(f"\nQuestion {i} Answer:")
            print(f"📝 {answer}")
            
    except Exception as e:
        print(f"❌ Sample query failed: {e}")
    
    # Test custom insurance query
    print("\n3. Testing Custom Insurance Query...")
    try:
        custom_questions = [
            "46-year-old male, knee surgery in Pune, 3-month-old insurance policy",
            "Does this policy cover knee surgery, and what are the conditions?",
            "What is the coverage amount for surgical procedures?"
        ]
        
        result = await tester.test_custom_query(
            "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D",
            custom_questions
        )
        
        print(f"✅ Processed custom queries")
        
        # Print answers
        answers = result.get('answers', [])
        for i, (question, answer) in enumerate(zip(custom_questions, answers), 1):
            print(f"\nCustom Query {i}:")
            print(f"❓ {question}")
            print(f"📝 {answer}")
            
    except Exception as e:
        print(f"❌ Custom query failed: {e}")
    
    # Get system stats
    print("\n4. Getting System Statistics...")
    try:
        stats = await tester.get_stats()
        print(f"✅ System Stats:")
        print(json.dumps(stats, indent=2))
    except Exception as e:
        print(f"❌ Stats retrieval failed: {e}")
    
    print("\n" + "=" * 50)
    print("🎉 Testing completed!")


if __name__ == "__main__":
    asyncio.run(main())
