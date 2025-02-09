import subprocess
import sys
import time


SCRIPTS = [
    #"decision_sell.py", #turn or off depending on the user
    #"decision_buy.py", #turn or off depending on the user
    "final_decision.py", #turn or off depending on the user
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

def main():
    total_scripts = len(SCRIPTS)
    
    print_colored(f"\nStarting execution of {total_scripts} scripts...\n", "blue")
    
    for index, script in enumerate(SCRIPTS, start=1):
        print_colored(f"[{index}/{total_scripts}] Executing: {script}", "yellow")
        start_time = time.time()
        
        try:
            subprocess.check_call([sys.executable, script])
            execution_time = time.time() - start_time
            print_colored(f"✅ Completed: {script} (Time: {execution_time:.2f} seconds)", "green")
        except subprocess.CalledProcessError as e:
            print_colored(f"❌ Error in {script}: {e}", "red")
            sys.exit(1)
        
        if index < total_scripts:
            print_colored(f"\nMoving to next script...\n", "blue")
    
    print_colored("\nAll scripts executed successfully! 🎉", "green")

if __name__ == "__main__":
    main()
