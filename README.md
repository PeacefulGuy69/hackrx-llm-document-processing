# LLM Document Processing System

## Overview

This is an intelligent document processing system built for the HackRx 6.0 hackathon. The system uses Large Language Models (LLMs) to process natural language queries and retrieve relevant information from unstructured documents such as policy documents, contracts, and emails.

## Features

- **Semantic Document Processing**: Extracts and chunks documents (PDF, DOCX, emails) for optimal processing
- **Intelligent Query Parsing**: Converts natural language queries into structured format
- **Vector-based Search**: Uses FAISS for fast semantic similarity search
- **Decision Engine**: Makes intelligent decisions based on policy rules and retrieved clauses
- **RESTful API**: FastAPI-based API with proper authentication and documentation
- **Explainable AI**: Provides detailed justifications for all decisions with clause references

## Architecture

```
Input Documents → Document Processor → Vector Store (FAISS)
                                           ↓
Query → Query Parser → Semantic Search → Decision Engine → Structured Response
```

## Components

1. **Document Processor** (`document_processor.py`): Handles PDF, DOCX, and email processing
2. **Query Parser** (`query_parser.py`): Extracts structured information from natural language
3. **Vector Store** (`vector_store.py`): FAISS-based semantic search with OpenAI embeddings
4. **Decision Engine** (`decision_engine.py`): Makes policy-based decisions using LLM reasoning
5. **Main Processor** (`llm_processor.py`): Orchestrates the entire pipeline
6. **API** (`main.py`): FastAPI server with authentication and endpoints

## Quick Start

### 1. Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration

Create a `.env` file from the example:

```bash
cp .env.example .env
```

Edit `.env` and add your OpenAI API key:

```env
OPENAI_API_KEY=your_openai_api_key_here
BEARER_TOKEN=60359a637b23864b320999e8d98517f239970ee339c266bde110414ce8fb9ed1
```

### 3. Run the Server

```bash
python main.py
```

The server will start at `http://localhost:8000`

### 4. Test the API

```bash
python test_api.py
```

## API Usage

### Main Endpoint

**POST** `/api/v1/hackrx/run`

```json
{
    "documents": "https://example.com/policy.pdf",
    "questions": [
        "What is the grace period for premium payment?",
        "Does this policy cover knee surgery?"
    ]
}
```

**Response:**

```json
{
    "answers": [
        "A grace period of thirty days is provided for premium payment after the due date.",
        "Yes, knee surgery is covered under the policy with specific conditions."
    ]
}
```

### Authentication

All API requests require a Bearer token:

```bash
Authorization: Bearer 60359a637b23864b320999e8d98517f239970ee339c266bde110414ce8fb9ed1
```

## Sample Queries

The system handles various types of insurance and policy queries:

### Coverage Queries
- "46-year-old male, knee surgery in Pune, 3-month-old insurance policy"
- "Does this policy cover maternity expenses?"
- "What is the waiting period for cataract surgery?"

### Policy Information
- "What is the grace period for premium payment?"
- "What is the No Claim Discount offered?"
- "Are there any sub-limits on room rent?"

### Claims Processing
- "Medical expenses for organ donor coverage"
- "AYUSH treatment coverage extent"
- "Preventive health check-up benefits"

## System Features

### Document Processing
- **PDF Processing**: Extracts text from PDF documents with page-wise organization
- **DOCX Support**: Handles Word documents with proper text extraction
- **Email Processing**: Parses email content including headers and body
- **Smart Chunking**: Creates optimal chunks with token counting and overlap

### Query Understanding
- **Entity Extraction**: Identifies age, gender, procedures, locations, amounts
- **Query Classification**: Categorizes queries as coverage, claims, or policy
- **Context Enhancement**: Uses both LLM and regex-based extraction

### Semantic Search
- **Vector Embeddings**: Uses OpenAI's text-embedding-3-small model
- **FAISS Integration**: Fast similarity search with cosine similarity
- **Relevance Scoring**: Returns most relevant clauses with confidence scores

### Decision Making
- **LLM Reasoning**: Uses GPT-4 for intelligent decision making
- **Rule-based Fallback**: Provides backup logic for reliability
- **Confidence Scoring**: Assigns confidence levels to decisions
- **Clause References**: Maps decisions to specific document clauses

## Technical Specifications

### Models Used
- **LLM**: GPT-4 Turbo Preview for reasoning and decision making
- **Embeddings**: OpenAI text-embedding-3-small (1536 dimensions)
- **Fallback Embeddings**: Sentence-Transformers all-MiniLM-L6-v2

### Performance Optimization
- **Document Caching**: In-memory caching of processed documents
- **Vector Store Persistence**: FAISS index saved to disk
- **Async Processing**: Full async/await implementation
- **Token Optimization**: Smart chunking to minimize token usage

### Security & Reliability
- **Bearer Token Authentication**: Secure API access
- **Error Handling**: Comprehensive exception handling
- **Health Checks**: System monitoring endpoints
- **Logging**: Detailed logging for debugging and monitoring

## Configuration Options

Key configuration parameters in `.env`:

```env
# LLM Configuration
LLM_MODEL=gpt-4-turbo-preview
EMBEDDING_MODEL=text-embedding-3-small
MAX_TOKENS=4000
TEMPERATURE=0.1

# Vector Database
VECTOR_DIMENSION=1536
TOP_K_RESULTS=10

# Document Processing
MAX_CHUNK_SIZE=1000
CHUNK_OVERLAP=200
MAX_FILE_SIZE_MB=50
```

## API Endpoints

### Core Endpoints
- `POST /api/v1/hackrx/run` - Main query processing
- `GET /health` - Health check
- `GET /api/v1/stats` - System statistics

### Management Endpoints
- `POST /api/v1/document/process` - Process individual documents
- `POST /api/v1/cache/clear` - Clear system cache

## Error Handling

The system provides comprehensive error handling:

- **Document Processing Errors**: Graceful handling of malformed documents
- **API Errors**: Proper HTTP status codes and error messages
- **LLM Failures**: Fallback to rule-based systems
- **Authentication Errors**: Clear authorization failure messages

## Monitoring & Debugging

### Logging
- Structured logging with timestamps and log levels
- Request/response logging for API calls
- Performance metrics (processing times)
- Error tracking with stack traces

### Statistics
- Document processing metrics
- Vector store statistics
- Cache hit rates
- Query processing times

## Development

### Project Structure
```
├── main.py                 # FastAPI application
├── config.py              # Configuration management
├── models.py              # Pydantic data models
├── document_processor.py   # Document processing logic
├── query_parser.py        # Query parsing and structuring
├── vector_store.py        # FAISS vector store management
├── decision_engine.py     # Decision making logic
├── llm_processor.py       # Main orchestrator
├── test_api.py           # API testing script
├── requirements.txt       # Python dependencies
└── .env.example          # Environment configuration template
```

### Adding New Features

1. **New Document Types**: Extend `document_processor.py`
2. **Query Types**: Update `query_parser.py` and `models.py`
3. **Decision Rules**: Modify `decision_engine.py`
4. **API Endpoints**: Add to `main.py`

## Deployment

### Local Development
```bash
python main.py
```

### Production Deployment
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```

### Docker Deployment
```dockerfile
FROM python:3.9-slim
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

## Evaluation Metrics

The system is optimized for:

- **Accuracy**: Precise query understanding and clause matching
- **Token Efficiency**: Optimized LLM token usage
- **Latency**: Fast response times with caching
- **Reusability**: Modular, extensible architecture
- **Explainability**: Clear decision reasoning and clause traceability

## License

This project is developed for the HackRx 6.0 hackathon.

## Contributing

This is a hackathon project. For questions or issues, please check the logging output and system statistics endpoints.
