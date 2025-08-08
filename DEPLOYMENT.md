# Deployment Guide

This application can be deployed on multiple platforms. Choose the one that best fits your needs.

## Prerequisites

1. Get a Google Gemini API key from [Google AI Studio](https://aistudio.google.com/app/apikey)
2. Set the `GEMINI_API_KEY` environment variable on your chosen platform

## Quick Deploy Options

### 1. Heroku
[![Deploy](https://www.herokucdn.com/deploy/button.svg)](https://heroku.com/deploy)

```bash
# Using Heroku CLI
heroku create your-app-name
heroku config:set GEMINI_API_KEY=your_api_key_here
git push heroku main
```

### 2. Railway
[![Deploy on Railway](https://railway.app/button.svg)](https://railway.app/new/template/python-fastapi)

1. Connect your GitHub repository
2. Set environment variable: `GEMINI_API_KEY`
3. Deploy automatically

### 3. Render
[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy)

1. Connect your GitHub repository
2. Use `render.yaml` configuration
3. Set environment variable: `GEMINI_API_KEY`

### 4. Vercel
```bash
npm i -g vercel
vercel --prod
# Set GEMINI_API_KEY in Vercel dashboard
```

### 5. DigitalOcean App Platform
1. Use the `.do/app.yaml` configuration
2. Set `GEMINI_API_KEY` as a secret environment variable

### 6. Netlify Functions
```bash
npm install -g netlify-cli
netlify deploy --prod
# Set GEMINI_API_KEY in Netlify dashboard
```

## Docker Deployment

### Local Docker
```bash
docker build -t hackrx-app .
docker run -p 8000:8000 -e GEMINI_API_KEY=your_key_here hackrx-app
```

### Docker Compose
```bash
# Create .env file with GEMINI_API_KEY
docker-compose up -d
```

### AWS ECS/EKS
Use the Dockerfile with your preferred container orchestration platform.

### Google Cloud Run
```bash
gcloud run deploy hackrx-app --source . --platform managed --region us-central1
```

### Azure Container Instances
```bash
az container create --resource-group myResourceGroup --name hackrx-app --image your-image --ports 8000
```

## Environment Variables

### Required
- `GEMINI_API_KEY`: Your Google Gemini API key

### Optional
- `HOST`: Server host (default: 0.0.0.0)
- `PORT`: Server port (default: 8000)
- `DEBUG`: Debug mode (default: false)
- `BEARER_TOKEN`: API authentication token
- `MAX_TOKENS`: Maximum tokens for LLM (default: 4000)
- `TEMPERATURE`: LLM temperature (default: 0.1)

## API Endpoints

After deployment, your webhook URL will be:
```
https://your-deployed-app.com/hackrx/run
```

### Available Endpoints:
- `GET /` - Health check
- `GET /health` - Detailed health check
- `POST /hackrx/run` - Main processing endpoint
- `POST /api/v1/document/process` - Document processing
- `GET /api/v1/stats` - System statistics

## Testing Your Deployment

```bash
curl -X POST "https://your-deployed-app.com/hackrx/run" \
  -H "Authorization: Bearer 60359a637b23864b320999e8d98517f239970ee339c266bde110414ce8fb9ed1" \
  -H "Content-Type: application/json" \
  -d '{
    "questions": ["What is this document about?"],
    "documents": ["https://example.com/sample.pdf"]
  }'
```

## Troubleshooting

1. **CORS Issues**: The app allows all origins by default
2. **Memory Issues**: Consider upgrading to a higher tier plan
3. **Cold Starts**: First request might be slower on serverless platforms
4. **File Size Limits**: Maximum 50MB per document by default

## Performance Optimization

- Use persistent storage for vector store in production
- Enable caching for frequently accessed documents
- Consider using Redis for distributed caching
- Monitor memory usage and scale accordingly
