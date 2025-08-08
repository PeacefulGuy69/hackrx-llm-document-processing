import sys
import os

def handler(event, context):
    """Minimal test handler for debugging"""
    return {
        "statusCode": 200,
        "headers": {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "Content-Type",
            "Access-Control-Allow-Methods": "GET, POST, OPTIONS"
        },
        "body": '{"message": "Hello from Netlify Functions!", "event": "' + str(event.get('path', '/')) + '"}'
    }
