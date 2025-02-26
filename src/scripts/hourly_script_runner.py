import subprocess
import sys
import time
import asyncio
import schedule
from datetime import datetime, timedelta

SCRIPTS = [
    "ExportDataFromMT.py",
    "ExportDataFromMT_dokko.py",
    "decision_sell.py",
    "decision_buy.py",
    "final_decision.py",
    "api_conn3.py",
    "decision_sell_dokko.py",
    "decision_buy_dokko.py",
    "final_decision_dokko.py",
    "api_conn3_dokko.py",
]

def print_colored(text, color):
    colors = {
        "green": "\033[92m",
        "blue": "\033[94m",
        "yellow": "\033[93m",
        "red": "\033[91m",
        "reset": "\033[0m"
    }
    print(f"{colors[color]}{text}{colors['reset']}")

async def run_script(script):
    print_colored(f"Executing: {script}", "yellow")
    start_time = time.time()
    
    try:
        process = await asyncio.create_subprocess_exec(
            sys.executable, script,
            #stdout=asyncio.subprocess.PIPE,
            #stderr=asyncio.subprocess.PIPE
        )
        await process.communicate()
        
        if process.returncode == 0:
            execution_time = time.time() - start_time
            print_colored(f"✅ Completed: {script} (Time: {execution_time:.2f} seconds)", "green")
        else:
            print_colored(f"❌ Error in {script}", "red")
    except Exception as e:
        print_colored(f"❌ Error in {script}: {e}", "red")

async def run_scripts():
    total_scripts = len(SCRIPTS)
    print_colored(f"\nStarting execution of {total_scripts} scripts...\n", "blue")
    
    for script in SCRIPTS:
        await run_script(script)
        await asyncio.sleep(1)  # 1-second delay between scripts
    
    print_colored("\nAll scripts executed! 🎉", "green")

async def schedule_runner():
    while True:
        now = datetime.now()
        next_run = now.replace(minute=55, second=0, microsecond=0)
        if now.minute >= 55:
            next_run += timedelta(hours=1)
        
        wait_time = (next_run - now).total_seconds()
        
        print_colored(f"Next run scheduled at: {next_run.strftime('%H:%M')}", "blue")
        await asyncio.sleep(wait_time)
        
        print_colored("Starting scheduled run", "blue")
        await run_scripts()

async def main():
    print_colored("Script scheduler started. Will run at 55 minutes past each hour.", "blue")
    await schedule_runner()



if __name__ == "__main__":
    asyncio.run(main())
