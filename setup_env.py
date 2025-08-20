#!/usr/bin/env python3
"""
Environment Setup Script for RAG API
This script helps you set up the required environment variables.
"""

import os
import sys

def create_env_file():
    """Create a .env file with placeholder values."""
    env_content = """# Environment variables for the RAG API
# Replace the placeholder values with your actual API keys

# Your API key for accessing the service
API_KEY=your_api_key_here

# Mistral AI API key for LLM operations
MISTRAL_API_KEY=your_mistral_api_key_here

# Hugging Face API key for embeddings and reranking
HUGGINGFACE_API_KEY=your_huggingface_api_key_here
"""
    
    try:
        with open('.env', 'w') as f:
            f.write(env_content)
        print("✓ Created .env file with placeholder values")
        print("  Please edit .env and add your actual API keys")
        return True
    except Exception as e:
        print(f"✗ Failed to create .env file: {e}")
        return False

def check_env_vars():
    """Check if required environment variables are set."""
    required_vars = ['API_KEY', 'MISTRAL_API_KEY', 'HUGGINGFACE_API_KEY']
    missing_vars = []
    
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        print(f"✗ Missing environment variables: {', '.join(missing_vars)}")
        return False
    else:
        print("✓ All required environment variables are set")
        return True

def main():
    print("RAG API Environment Setup")
    print("=" * 40)
    
    # Check if .env file exists
    if os.path.exists('.env'):
        print("✓ .env file already exists")
        if check_env_vars():
            print("\nYou're all set! You can now run the application.")
            return
        else:
            print("\nPlease update your .env file with the actual API keys.")
            return
    
    # Create .env file
    print("Creating .env file...")
    if create_env_file():
        print("\nNext steps:")
        print("1. Edit the .env file and add your actual API keys")
        print("2. Run the application with: uvicorn main:app --host 0.0.0.0 --port 8000")
    else:
        print("\nFailed to create .env file. Please create it manually.")

if __name__ == "__main__":
    main()
