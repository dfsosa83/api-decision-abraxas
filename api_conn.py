import requests
import json

API_ENDPOINT = "https://api.perplexity.ai/chat/completions"
API_KEY = "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"

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

data = {
    "model": "pplx-7b-online",
    "messages": [
        {"role": "system", "content": "You are a quantum forex trading expert."},
        {"role": "user", "content": "Validate this forex prediction: Buy EUR/USD at 1.0750"}
    ]
}
