import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class Config:
    # API Configuration
    HOST = os.getenv("HOST", "0.0.0.0")
    PORT = int(os.getenv("PORT", 8000))
    DEBUG = os.getenv("DEBUG", "False").lower() == "true"
    API_PREFIX = os.getenv("API_PREFIX", "/api/v1")
    
    # Authentication
    BEARER_TOKEN = os.getenv("BEARER_TOKEN", "60359a637b23864b320999e8d98517f239970ee339c266bde110414ce8fb9ed1")
    
    # OpenAI Configuration
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4-turbo-preview")
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    MAX_TOKENS = int(os.getenv("MAX_TOKENS", 4000))
    TEMPERATURE = float(os.getenv("TEMPERATURE", 0.1))
    
    # Vector Database Configuration
    FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH", "./vector_store/faiss_index")
    VECTOR_DIMENSION = int(os.getenv("VECTOR_DIMENSION", 1536))
    TOP_K_RESULTS = int(os.getenv("TOP_K_RESULTS", 10))
    
    # Document Processing Configuration
    MAX_CHUNK_SIZE = int(os.getenv("MAX_CHUNK_SIZE", 1000))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 200))
    MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", 50))
    
    # Application Configuration
    APP_VERSION = "1.0.0"
    APP_NAME = "LLM Document Processing System"
    APP_DESCRIPTION = "Intelligent query-retrieval system for document processing"
    
    @classmethod
    def validate_config(cls):
        """Validate required configuration"""
        if not cls.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY environment variable is required")
        
        return True


# Create global config instance
config = Config()

# Validate configuration on import
config.validate_config()
