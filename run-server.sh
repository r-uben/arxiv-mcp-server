#!/bin/bash
cd "$(dirname "$0")"

# Load Mistral API key if the function exists (from ~/.zshrc)
if command -v load_mistral_key >/dev/null 2>&1; then
    load_mistral_key
    echo "Mistral API key loaded from secrets"
fi

# Start the server
poetry run arxiv-mcp-server