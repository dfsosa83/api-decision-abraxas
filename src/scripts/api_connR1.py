import requests
import pandas as pd
import os
import re

API_ENDPOINT = "https://api.perplexity.ai/chat/completions"
API_KEY = os.getenv("PPLX_API_KEY")  # Recommended: Use environment variable

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

def load_trading_data():
    """Load and validate trading decision data"""
    path = os.path.join("C:/Users/david/OneDrive/Documents/api-decision-abraxas/data/processed", 
                       "agent_information.csv")
    df = pd.read_csv(path)
    return df[df['action'].isin(['sell', 'buy'])].copy()

def create_validation_prompt(action: str, pair: str) -> dict:
    """Construct structured validation prompt with contradiction checks"""
    analysis_template = """Analyze this forex prediction for {pair} in this order:
    1. Recent News: Check tradingview.com/news, investing.com/analysis, dailyforex.com
    2. Market Sentiment: Evaluate dailyfx.com/sentiment and forexfactory.com/market
    3. Contradiction Check: Compare technical indicators vs fundamental analysis
    4. Risk Assessment: Evaluate volatility and liquidity factors

    Required format:
    [1-2 sentence summary] 
    CONFIRMATION RATIONALE: [bullet points]
    CONTRADICTION CHECK: [explicit contradictions found]
    FINAL DECISION: [STRICT FORMAT: CONFIRM/REJECT {action} {pair}]"""

    return {
        "model": "sonar-reasoning-pro",
        "messages": [
            {
                "role": "system",
                "content": "You are a quantum forex analyst specializing in multi-dimensional market validation. Strictly follow the response format."
            },
            {
                "role": "user",
                "content": analysis_template.format(pair=pair, action=action.upper())
            }
        ]
    }

def parse_decision(response: str) -> str:
    """Extract final decision from API response with validation"""
    decision_match = re.search(r'FINAL DECISION:\s*(CONFIRM|REJECT)\s+(\w+/\w+)', 
                              response, re.IGNORECASE)
    if decision_match:
        return f"{decision_match.group(1).upper()} {decision_match.group(2)}"
    return "INDETERMINATE"

# Execution flow
if __name__ == "__main__":
    trades = load_trading_data()
    
    for _, row in trades.iterrows():
        try:
            response = requests.post(
                API_ENDPOINT,
                headers=headers,
                json=create_validation_prompt(row['final_decision'], row['currency_pair']),
                timeout=15
            ).json()['choices'][0]['message']['content']
            
            print(f"Validation for {row['currency_pair']} {row['final_decision'].upper()}:")
            print(f"API Response:\n{response}")
            print(f"Parsed Decision: {parse_decision(response)}\n")
            
        except Exception as e:
            print(f"Error processing {row['currency_pair']}: {str(e)}")
