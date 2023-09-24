import subprocess
from datetime import date
import os

def run_script(script_path, log_file):
    print(f"Running {script_path}...")
    try:
        with open(log_file, 'w') as log:
            subprocess.run(['python', script_path], check=True, stdout=log, stderr=subprocess.STDOUT)
        print(f"{script_path} completed successfully. Log saved to {log_file}")
    except subprocess.CalledProcessError as e:
        print(f"Error running {script_path}: {e}")

# Create a directory for logs with today's date
log_folder = os.path.join('logs', date.today().strftime('%Y-%m-%d'))
os.makedirs(log_folder, exist_ok=True)

# List of scripts to run in order along with log file names
script_list = [
    ('seeds/check_new_dates.py', 'check_new_dates_log.txt'),
    ('download_json.py', 'download_json_log.txt'),
    ('processed_output/attendance.py', 'attendance_log.txt'),
    ('processed_output/member.py', 'member_log.txt')
]

for script, log_file in script_list:
    log_file_path = os.path.join(log_folder, log_file)
    run_script(script, log_file_path)
