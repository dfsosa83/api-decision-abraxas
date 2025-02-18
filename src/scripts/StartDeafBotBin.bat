rem Step 1: Activate Miniconda, use the actual path where your Miniconda/Ananconda is installed.
call "C:\Users\mauri\miniconda3\Scripts\activate.bat" "C:\Users\Administrator\miniconda3"

rem Step 2: Activate Conda environment. 
call conda activate deaf

rem Step 3: Change directory to the desired folder.
cd /d "C:\Users\mauri\Desktop\DeafBot"

rem Step 4: Run the Python script.
rem python main.py
call flask --app predictBin run --port 5000

pause
