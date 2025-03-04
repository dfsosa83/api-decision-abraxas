import requests
import pandas as pd
import os
import re
import csv
import json
from datetime import datetime, timedelta
import pytz

API_ENDPOINT = "https://api.perplexity.ai/chat/completions"
API_KEY = "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"

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

path = "C:/Users/david/OneDrive/Documents/api-decision-abraxas/data//processed/last_final_decision.csv"
decisions_file = "C:/Users/david/OneDrive/Documents/api-decision-abraxas/data/decisions/decisions.csv"
trade_execution_file = "C:/Users/david/OneDrive/Documents/api-decision-abraxas/data/decisions/trade_executions.json"

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

Justification: [2-3 sentences explaining the final decision]

If the FINAL DECISION is CONFIRM or REJECT(if is REJECT, just add a warning to trader), please provide the following, remember brackets[] in **Trade Execution** answer, keep with brackects and numbers no letters need, i need to conserve format to save the data:

**Trade Execution**
- Entry: [Current price range]
- Stop Loss: [Price level]
- Take Profit: [Price level]

Ensure that the Stop Loss and Take Profit levels are based on key support/resistance levels and maintain a favorable risk-reward ratio."""

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
    match = re.search(r'Justification:(.*?)(?:\n\n|\Z)', response, re.DOTALL)
    if match:
        return match.group(1).strip()
    return "No justification provided"

def extract_trade_execution(response):
    pattern = r'(?i)(Entry|Stop Loss|Take Profit)\s*:\s*\[?([\d\.-]+(?:-[\d\.-]+)?)\]?'
    matches = re.findall(pattern, response)
    result = {}
    for key, value in matches:
        key = key.title().replace(' ', '_')
        result[key] = value.strip('[]')
    return result

os.makedirs(os.path.dirname(decisions_file), exist_ok=True)
os.makedirs(os.path.dirname(trade_execution_file), exist_ok=True)

def append_decision(decision):
    df = pd.DataFrame([decision])
    df.to_csv(decisions_file, mode='a', header=not os.path.exists(decisions_file), index=False)

def append_trade_execution(decision, trade_execution):
    try:
        required_keys = ['Entry', 'Stop_Loss', 'Take_Profit']
        if not all(k in trade_execution for k in required_keys):
            raise ValueError("Missing trade execution parameters")
        
        with open(trade_execution_file, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=required_keys)
            if f.tell() == 0:
                writer.writeheader()
            writer.writerow(trade_execution)
        print(f"Data written to {trade_execution_file}")
    except Exception as e:
        print(f"Critical write error: {str(e)}")


df = pd.read_csv(path).tail(1)
print(df)

def save_full_response(response_text):
    log_dir = "C:/Users/david/OneDrive/Documents/api-decision-abraxas/data/logs/"
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(log_dir, "agent_responses.log")
    
    try:
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(f"\n{'='*80}\n")
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(response_text)
            f.write(f"\n{'='*80}\n")
        print(f"Full response saved to: {log_file}")
    except Exception as e:
        print(f"Failed to save response: {str(e)}")

if df['final_decision'].iloc[0] in ['sell', 'buy']:
    action = df['final_decision'].iloc[0]
    currency_pair = df['currency_pair'].iloc[0]
    
    local_tz = pytz.timezone('America/Panama')
    market_tz = pytz.FixedOffset(5 * 60)  # GMT+5
    
    current_time = datetime.now(pytz.utc)
    local_time = current_time.astimezone(local_tz)
    market_time = current_time.astimezone(market_tz)
    
    result = query_perplexity(action, currency_pair, local_time, market_time)
    save_full_response(result)
    print(f"Raw API Response:\n{result}\n")
    
    print(f"Query: {action} {currency_pair}")
    print(f"Local Time (Panama): {local_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    trade_execution = extract_trade_execution(result)
    print(f"Parsed Execution Data: {trade_execution}")
    
    final_decision = extract_final_decision(result)
    justification = extract_justification(result)
    
    decision = {
    'timestamp_local': local_time.strftime("%Y-%m-%d %H:%M:%S"),
    'timestamp_market': market_time.strftime("%Y-%m-%d %H:%M:%S"),
    'currency_pair': currency_pair,
    'model_suggestion': action,
    'agent_decision': final_decision,
    'justification': justification,
    'entry': trade_execution.get('Entry', ''),
    'stop_loss': trade_execution.get('Stop_Loss', ''),
    'take_profit': trade_execution.get('Take_Profit', '')
}

    append_decision(decision)
    print(f"Decision appended to: {decisions_file}")

    if final_decision in ['CONFIRM','REJECT']:
        try:
            append_trade_execution(decision, trade_execution)
        except ValueError as e:
            print(f"Invalid Trade Data: {str(e)}")
        with open('failed_responses.log', 'a') as f:
            f.write(f"{datetime.now()} | {result}\n\n")
else:
    print("No valid action (buy/sell) found in the last row of last_final_decision.csv")
