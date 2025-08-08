import sys
import os

# Add the parent directories to the path
current_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, parent_dir)

try:
    from main import app
    from mangum import Mangum
    
    # Create the handler with proper configuration for Netlify
    handler = Mangum(
        app, 
        lifespan="off",
        api_gateway_base_path=None,
        text_mime_types=[
            "application/json",
            "application/javascript",
            "application/xml",
            "application/vnd.api+json",
            "text/html",
            "text/plain",
            "text/css",
        ]
    )
    
except Exception as e:
    import json
    print(f"Error importing or configuring app: {e}")
    
    # Fallback handler
    def handler(event, context):
        return {
            "statusCode": 500,
            "headers": {
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*"
            },
            "body": json.dumps({
                "error": f"Configuration error: {str(e)}",
                "path": event.get('path', 'unknown'),
                "method": event.get('httpMethod', 'unknown')
            })
        }
