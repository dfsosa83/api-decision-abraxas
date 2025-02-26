import requests
import pandas as pd
import os
import re
import csv
from datetime import datetime, timedelta
import pytz

API_ENDPOINT = "https://api.perplexity.ai/chat/completions"
API_KEY = "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"

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

path = "C:/Users/david/OneDrive/Documents/api-decision-abraxas/data//processed/last_final_decision_dokkodo.csv"
decisions_file = "C:/Users/david/OneDrive/Documents/api-decision-abraxas/data/decisions/decisions_dokkodo.csv"

df = pd.read_csv(path)
print(df)

filtered_df = df[df['final_decision'].isin(['sell', 'buy'])]

def query_perplexity(action, currency_pair, local_time, market_time):
    analysis_template = f"""Validate this forex prediction for {currency_pair} to {action} in the next hour:

Current market time (GMT+5): {market_time.strftime("%Y-%m-%d %H:%M:%S")}
Current local time (Panama): {local_time.strftime("%Y-%m-%d %H:%M:%S")}

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
   - validate this web site too: https://www.fxempire.com/

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
    
    response = requests.post(API_ENDPOINT, headers=headers, json=data,verify=False)
    
    if response.status_code == 200:
        return response.json()['choices'][0]['message']['content']
    else:
        return f"Error: {response.status_code}, {response.text}"

def extract_final_decision(response):
    match = re.search(r'FINAL DECISION: (CONFIRM|REJECT)', response)
    if match:
        return match.group(1)
    return "UNKNOWN"

def extract_justification(response):
    match = re.search(r'\*\*Justification\*\*:(.*)', response, re.DOTALL)
    if match:
        return match.group(1).strip()
    return "No justification provided"

os.makedirs(os.path.dirname(decisions_file), exist_ok=True)

def append_decision(decision):
    df = pd.DataFrame([decision])
    df.to_csv(decisions_file, mode='a', header=not os.path.exists(decisions_file), index=False)

df = pd.read_csv(path).tail(1)
print(df)

if df['final_decision'].iloc[0] in ['sell', 'buy']:
    action = df['final_decision'].iloc[0]
    currency_pair = df['currency_pair'].iloc[0]
    
    local_tz = pytz.timezone('America/Panama')
    market_tz = pytz.FixedOffset(5 * 60)  # GMT+5
    
    current_time = datetime.now(pytz.utc)
    local_time = current_time.astimezone(local_tz)
    market_time = current_time.astimezone(market_tz)
    
    result = query_perplexity(action, currency_pair, local_time, market_time)
    print(f"Query: {action} {currency_pair}")
    print(f"Local Time (Panama): {local_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Market Time (GMT+5): {market_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Response: {result}\n")
    
    final_decision = extract_final_decision(result)
    justification = extract_justification(result)
    
    decision = {
        'timestamp_local': local_time.strftime("%Y-%m-%d %H:%M:%S"),
        'timestamp_market': market_time.strftime("%Y-%m-%d %H:%M:%S"),
        'currency_pair': currency_pair,
        'model_suggestion': action,
        'agent_decision': final_decision,
        'justification': justification
    }
    
    append_decision(decision)
    
    print(f"Decision appended to: {decisions_file}")
else:
    print("No valid action (buy/sell) found in the last row of last_final_decision_dokkodo.csv")
