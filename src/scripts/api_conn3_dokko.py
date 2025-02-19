import requests
import pandas as pd
import os
import re
import csv
from datetime import datetime

API_ENDPOINT = "https://api.perplexity.ai/chat/completions"
API_KEY = "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"
#API_KEY = os.getenv("PPLX_API_KEY")

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

# Path to last final decision
path = "C:/Users/david/OneDrive/Documents/api-decision-abraxas/data//processed/last_final_decision_dokkodo.csv"

# Path to the persistent decisions file
decisions_file = "C:/Users/david/OneDrive/Documents/api-decision-abraxas/data/decisions/decisions_dokkodo.csv"

# Load data
df = pd.read_csv(path)
print(df)

# Filter for 'sell' or 'buy' values in the 'final_decision' column
filtered_df = df[df['final_decision'].isin(['sell', 'buy'])]

def query_perplexity(action, currency_pair):
    analysis_template = f"""Validate this forex prediction for {currency_pair} to {action} in the next hour:

1. Technical Analysis (1-hour timeframe):
   - Analyze key indicators: Moving Averages, RSI, MACD, Bollinger Bands
   - Identify support/resistance levels
   - Evaluate chart patterns and price action

2. Fundamental Analysis:
   - Review recent economic data releases affecting {currency_pair}
   - Check for upcoming high-impact events in the next 24 hours
   - Assess central bank policies and statements

3. Market Sentiment:
   - Evaluate retail trader positioning
   - Analyze institutional sentiment and COT data
   - Review overall market risk appetite

4. News Analysis:
   - Summarize relevant breaking news from reliable forex sources
   - Assess potential impact on {currency_pair}

5. Correlation Analysis:
   - Check correlations with related currency pairs
   - Evaluate impacts from commodity prices or stock market movements

Provide your analysis in the following format:
1. Summary: [2-3 sentence overview]
2. Technical Outlook: [Bullish/Bearish/Neutral with key points]
3. Fundamental Factors: [List key fundamental drivers]
4. Sentiment Analysis: [Current market sentiment with supporting data]
5. Risk Assessment: [Identify potential risks to the trade]
6. Contradiction Check: [Note any conflicting signals between technical and fundamental analysis]

FINAL DECISION: [CONFIRM/REJECT] {action.upper()} {currency_pair} for the next hour

Justification: [2-3 sentences explaining the final decision]"""

    data = {
        "model": "sonar-reasoning-pro",
        "messages": [
            {"role": "system", "content": "You are a quantum forex trading expert specializing in short-term market analysis and validation."},
            {"role": "user", "content": analysis_template}
        ]
    }
    
    response = requests.post(API_ENDPOINT, headers=headers, json=data)
    
    if response.status_code == 200:
        return response.json()['choices'][0]['message']['content']
    else:
        return f"Error: {response.status_code}, {response.text}"


# Function to extract FINAL DECISION
def extract_final_decision(response):
    match = re.search(r'FINAL DECISION: (CONFIRM|REJECT)', response)
    if match:
        return match.group(1)
    return "UNKNOWN"

# Create the decisions directory if it doesn't exist
os.makedirs(os.path.dirname(decisions_file), exist_ok=True)

# Function to append a decision to the CSV file
def append_decision(decision):
    df = pd.DataFrame([decision])
    df.to_csv(decisions_file, mode='a', header=not os.path.exists(decisions_file), index=False)

# Read only the last row from last_final_decision.csv
df = pd.read_csv(path).tail(1)
print(df)

# Check if the last row contains 'sell' or 'buy' in the 'final_decision' column
if df['final_decision'].iloc[0] in ['sell', 'buy']:
    action = df['final_decision'].iloc[0]
    currency_pair = df['currency_pair'].iloc[0]
    
    result = query_perplexity(action, currency_pair)
    print(f"Query: {action} {currency_pair}")
    print(f"Response: {result}\n")
    
    # Extract the FINAL DECISION
    final_decision = extract_final_decision(result)
    
    # Create the decision dictionary
    decision = {
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'currency_pair': currency_pair,
        'model_suggestion': action,
        'agent_decision': final_decision
    }
    
    # Append the decision to the CSV file
    append_decision(decision)
    
    print(f"Decision appended to: {decisions_file}")
else:
    print("No valid action (buy/sell) found in the last row of last_final_decision.csv")
    


