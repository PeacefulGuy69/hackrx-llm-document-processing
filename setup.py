import asyncio
import os
import json
from pathlib import Path


async def setup_system():
    """Setup and validate the LLM Document Processing System"""
    
    print("🚀 LLM Document Processing System Setup")
    print("=" * 50)
    
    # Check Python version
    import sys
    print(f"✅ Python version: {sys.version}")
    
    # Check if .env exists
    env_path = Path(".env")
    if not env_path.exists():
        print("⚠️  .env file not found, creating from template...")
        with open(".env.example", "r") as f:
            content = f.read()
        with open(".env", "w") as f:
            f.write(content)
        print("📝 Please edit .env file and add your OpenAI API key")
        return False
    
    # Load environment
    from dotenv import load_dotenv
    load_dotenv()
    
    # Check OpenAI API key
    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key or openai_key == "your_openai_api_key_here":
        print("❌ OpenAI API key not configured")
        print("   Please edit .env file and set OPENAI_API_KEY=your_key")
        return False
    
    print(f"✅ OpenAI API key configured (ends with: ...{openai_key[-4:]})")
    
    # Test OpenAI connection
    try:
        from openai import AsyncOpenAI
        client = AsyncOpenAI(api_key=openai_key)
        
        # Simple test
        response = await client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=10
        )
        print("✅ OpenAI API connection successful")
        
    except Exception as e:
        print(f"❌ OpenAI API connection failed: {str(e)}")
        return False
    
    # Check required packages
    required_packages = [
        "fastapi", "uvicorn", "openai", "faiss-cpu", 
        "sentence-transformers", "pymupdf", "python-docx"
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package.replace("-", "_"))
            print(f"✅ {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package} - not installed")
    
    if missing_packages:
        print(f"\n📦 Install missing packages: pip install {' '.join(missing_packages)}")
        return False
    
    # Create necessary directories
    os.makedirs("vector_store", exist_ok=True)
    print("✅ Vector store directory created")
    
    # Test system components
    print("\n🧪 Testing system components...")
    
    try:
        # Test document processor
        from document_processor import document_processor
        print("✅ Document processor imported")
        
        # Test query parser
        from query_parser import query_parser
        test_query = "46-year-old male, knee surgery in Pune"
        structured = await query_parser.parse_query(test_query)
        print(f"✅ Query parser: {structured.age} year old {structured.gender}")
        
        # Test vector store
        from vector_store import semantic_searcher
        stats = semantic_searcher.get_stats()
        print(f"✅ Vector store: {stats['total_chunks']} chunks")
        
        # Test decision engine
        from decision_engine import decision_engine
        print("✅ Decision engine imported")
        
    except Exception as e:
        print(f"❌ Component test failed: {str(e)}")
        return False
    
    print("\n🎉 System setup completed successfully!")
    print("\n📋 Next steps:")
    print("1. Run: python main.py (to start the server)")
    print("2. Test: python test_api.py (to test the API)")
    print("3. Access: http://localhost:8000/docs (for API documentation)")
    
    return True


def create_sample_env():
    """Create a sample .env file with instructions"""
    sample_env = """# LLM Document Processing System Configuration

# ⚠️ REQUIRED: Your OpenAI API Key
# Get your API key from: https://platform.openai.com/api-keys
OPENAI_API_KEY=your_openai_api_key_here

# Authentication token for the HackRx API (pre-configured)
BEARER_TOKEN=60359a637b23864b320999e8d98517f239970ee339c266bde110414ce8fb9ed1

# Server configuration
HOST=0.0.0.0
PORT=8000
DEBUG=True

# LLM Configuration
LLM_MODEL=gpt-4-turbo-preview
EMBEDDING_MODEL=text-embedding-3-small
MAX_TOKENS=4000
TEMPERATURE=0.1

# Vector Database Configuration
FAISS_INDEX_PATH=./vector_store/faiss_index
VECTOR_DIMENSION=1536
TOP_K_RESULTS=10

# Document Processing Configuration
MAX_CHUNK_SIZE=1000
CHUNK_OVERLAP=200
MAX_FILE_SIZE_MB=50

# API Configuration
API_PREFIX=/api/v1

# Instructions:
# 1. Replace 'your_openai_api_key_here' with your actual OpenAI API key
# 2. Save this file
# 3. Run: python setup.py
"""
    
    with open(".env.example", "w") as f:
        f.write(sample_env)
    
    print("📝 Updated .env.example with detailed instructions")


if __name__ == "__main__":
    # Update the .env.example file
    create_sample_env()
    
    # Run setup
    success = asyncio.run(setup_system())
    
    if not success:
        print("\n❌ Setup incomplete. Please fix the issues above and run again.")
        exit(1)
    else:
        print("\n✅ Setup completed successfully!")
        exit(0)
