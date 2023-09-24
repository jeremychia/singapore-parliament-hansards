from datetime import date
import pandas as pd
import os
import re

def get_source_file_path(csv_subfolder):
    '''
    This assumes that the Python Script
    is in one subfolder layer from the root.
    '''

    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_directory, _ = os.path.split(script_dir)

    return os.path.join(root_directory, csv_subfolder)

def get_attendance_file_names(source_file_path):
    '''
    Returns files that have "attendance.csv" in file name.
    '''

    file_names = os.listdir(source_file_path)
    return [file for file in file_names if "attendance.csv" in file]

def drop_present_files(csv_filename, attendance_files):

    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))

        files_processed_df = pd.read_csv(
            os.path.join(script_dir, csv_filename)
            )

        files_processed = files_processed_df['file_name'].tolist()
    except:
        files_processed = set()

    return list(sorted(set(attendance_files) - set(files_processed)))

def get_mp_name(x):

    if pd.notna(x) and 'SPEAKER' in x:
        temp = re.search(r'\(([^()]+)\(', x)
        if temp:
            match = re.sub(r'^(?:Mr|Mrs|Miss|Mdm|Ms|Dr|Prof)\s+', '', temp.group(1))
            return match
        else:
            return ''
    elif pd.notna(x):
        match = re.search(r'(?:Mr|Mrs|Miss|Mdm|Ms|Dr|Prof)\s+([\w\s-]+)', x)
        if match:
            return match.group(1)
        else:
            return ''
    else:
        return ''

def write_csv(csv_filename, df):
    
    if os.path.exists(csv_filename):
        df.to_csv(csv_filename,
                  mode='a', index=False,
                  header=False)
    else:
        df.to_csv(csv_filename,
                  mode='w', index=False)

    return 'Successful'

def log_files_processed(csv_filename, files_processed):

    script_dir = os.path.dirname(os.path.abspath(__file__))
    today_date = date.today().strftime('%Y-%m-%d')

    files_processed_df = pd.DataFrame({
            'file_name': files_processed,
            'processed_at': [today_date] * len(files_processed)
        })

    write_csv(csv_filename, files_processed_df)

    return 'Successful'


### Variables

csv_subfolder = 'code_output'
log_filename = 'log_attendance_files_read.csv'
attendance_filename = 'fact_attendance.csv'
script_dir = os.path.dirname(os.path.abspath(__file__))

### Main run here

attendance_files_dir = get_source_file_path(csv_subfolder)
attendance_files = get_attendance_file_names(attendance_files_dir)
to_process_files = drop_present_files(log_filename, attendance_files)

files_processed = []

for file_name in to_process_files:

    attendance_cur_df = pd.read_csv(
        os.path.join(attendance_files_dir, file_name)
        )
    attendance_cur_df['MP_Name'] = attendance_cur_df['MP_Name'].str.strip()
    attendance_cur_df['MP_Name_Title'] = attendance_cur_df['MP_Name']
    attendance_cur_df['MP_Name'] = attendance_cur_df['MP_Name'].apply(get_mp_name)
    
    write_csv(os.path.join(script_dir, attendance_filename)
              ,attendance_cur_df)

    files_processed.append(file_name)

    print(f'Processed: {file_name}')


log_files_processed(log_filename, files_processed)
    
    
    

