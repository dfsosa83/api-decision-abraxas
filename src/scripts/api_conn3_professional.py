"""
Forex Trading Decision Validation Script

This script validates forex trading decisions using the Perplexity AI API.
It processes trading decisions, queries the API for analysis, and logs the results.

Author: [David Sosa]
Date: [Current Date]
"""

import os
import re
import csv
import json
import requests
import pandas as pd
from datetime import datetime
import pytz

# API Configuration
API_ENDPOINT = "https://api.perplexity.ai/chat/completions"
API_KEY = "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"

# File Paths
BASE_PATH = "C:/Users/david/OneDrive/Documents/api-decision-abraxas/data"
DECISIONS_PATH = f"{BASE_PATH}/processed/last_final_decision.csv"
DECISIONS_OUTPUT = f"{BASE_PATH}/decisions/decisions.csv"
TRADE_EXECUTION_FILE = f"{BASE_PATH}/decisions/trade_executions.json"
LOG_DIR = f"{BASE_PATH}/logs/"

# Ensure necessary directories exist
os.makedirs(os.path.dirname(DECISIONS_OUTPUT), exist_ok=True)
os.makedirs(os.path.dirname(TRADE_EXECUTION_FILE), exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

def query_perplexity(action, currency_pair, local_time, market_time):
    """
    Query the Perplexity AI API for forex trading analysis.

    Args:
        action (str): The trading action (buy/sell).
        currency_pair (str): The currency pair to analyze.
        local_time (datetime): The current local time.
        market_time (datetime): The current market time.

    Returns:
        str: The API response content.
    """
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

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

    If the FINAL DECISION is CONFIRM or REJECT(if is REJECT, just add a warning to trader), please provide the following, remember brackets[] in **Trade Execution** answer, i need to conserve format to save the data:

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

    response = requests.post(API_ENDPOINT, headers=headers, json=data, verify=False)
    
    if response.status_code == 200:
        return response.json()['choices'][0]['message']['content']
    else:
        return f"Error: {response.status_code}, {response.text}"

def extract_final_decision(response):
    """Extract the final decision from the API response."""
    match = re.search(r'FINAL DECISION: (CONFIRM|REJECT)', response)
    return match.group(1) if match else "UNKNOWN"

def extract_justification(response):
    """Extract the justification from the API response."""
    match = re.search(r'Justification:(.*?)(?:\n\n|\Z)', response, re.DOTALL)
    return match.group(1).strip() if match else "No justification provided"

def extract_trade_execution(response):
    """Extract trade execution details from the API response."""
    pattern = r'(?i)(Entry|Stop Loss|Take Profit)\s*:\s*\[?([\d\.-]+(?:-[\d\.-]+)?)\]?'
    matches = re.findall(pattern, response)
    return {key.title().replace(' ', '_'): value.strip('[]') for key, value in matches}

def append_decision(decision):
    """Append a decision to the decisions CSV file."""
    df = pd.DataFrame([decision])
    df.to_csv(DECISIONS_OUTPUT, mode='a', header=not os.path.exists(DECISIONS_OUTPUT), index=False)

def append_trade_execution(decision, trade_execution):
    """Append trade execution details to the JSON file."""
    required_keys = ['Entry', 'Stop_Loss', 'Take_Profit']
    if not all(k in trade_execution for k in required_keys):
        raise ValueError("Missing trade execution parameters")

    with open(TRADE_EXECUTION_FILE, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=required_keys)
        if f.tell() == 0:
            writer.writeheader()
        writer.writerow(trade_execution)
    print(f"Data written to {TRADE_EXECUTION_FILE}")

def save_full_response(response_text):
    """Save the full API response to a log file."""
    log_file = os.path.join(LOG_DIR, "agent_responses.log")
    try:
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(f"\n{'='*80}\n")
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(response_text)
            f.write(f"\n{'='*80}\n")
        print(f"Full response saved to: {log_file}")
    except Exception as e:
        print(f"Failed to save response: {str(e)}")

def main():
    """Main function to process the latest trading decision and validate it."""
    df = pd.read_csv(DECISIONS_PATH).tail(1)
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
        print(f"Decision appended to: {DECISIONS_OUTPUT}")

        if final_decision in ['CONFIRM', 'REJECT']:
            try:
                append_trade_execution(decision, trade_execution)
            except ValueError as e:
                print(f"Invalid Trade Data: {str(e)}")
                with open('failed_responses.log', 'a') as f:
                    f.write(f"{datetime.now()} | {result}\n\n")
    else:
        print("No valid action (buy/sell) found in the last row of last_final_decision.csv")

if __name__ == "__main__":
    main()
