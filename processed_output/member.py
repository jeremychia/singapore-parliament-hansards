import pandas as pd
from datetime import date
import os

def summarise_attendance_information(df):

    return df.groupby('MP_Name').agg(
        earliest_sitting = pd.NamedAgg(
            column='Date',
            aggfunc='min'
            ),
        latest_sitting = pd.NamedAgg(
            column='Date',
            aggfunc='max'
            ),
        count_attendance = pd.NamedAgg(
            column='Attendance',
            aggfunc='sum'
            ),
        count_non_attendance=pd.NamedAgg(
            column='Attendance',
            aggfunc=lambda x: len(x) - x.sum()
            ),
        count_total_sittings_as_mp=pd.NamedAgg(
            column='Sitting_CID',
            aggfunc='count'
            )
        ).reset_index()

def write_csv(csv_filename, df):
    
    df.to_csv(csv_filename,
              mode='w', index=False)

    return 'Successful'

### Variables

script_dir = os.path.dirname(os.path.abspath(__file__))
attendance_file = 'fact_attendance.csv'
member_file = 'dim_member.csv'
today_date = date.today().strftime('%Y-%m-%d')

### Main run here

attendance_df = pd.read_csv(os.path.join(script_dir, attendance_file))

df = summarise_attendance_information(attendance_df)
write_csv(os.path.join(script_dir, member_file), df)
