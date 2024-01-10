import pandas as pd
from datetime import date
import os


def get_source_file_path(csv_subfolder):
    """
    This assumes that the Python Script
    is in one subfolder layer from the root.
    """

    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_directory, _ = os.path.split(script_dir)

    return os.path.join(root_directory, csv_subfolder)


def join_df(df1, df2, df2_name, join_on="MP_Name"):
    merged_df = pd.merge(df1, df2, on=join_on, how="left")

    mp_names_difference = set(df1[join_on]) - set(df2[join_on])

    for mp_name in mp_names_difference:
        print(f"Missing: {mp_name}, check {df2_name}.")

    return merged_df


def summarise_attendance_information(df):
    return (
        df.groupby("MP_Name")
        .agg(
            earliest_sitting=pd.NamedAgg(column="Date", aggfunc="min"),
            latest_sitting=pd.NamedAgg(column="Date", aggfunc="max"),
            count_attendance=pd.NamedAgg(column="Attendance", aggfunc="sum"),
            count_non_attendance=pd.NamedAgg(
                column="Attendance", aggfunc=lambda x: len(x) - x.sum()
            ),
            count_total_sittings_as_mp=pd.NamedAgg(
                column="Sitting_CID", aggfunc="count"
            ),
        )
        .reset_index()
    )


def write_csv(csv_filename, df):
    df.to_csv(csv_filename, mode="w", index=False)

    return "Successful"


### Variables

script_dir = os.path.dirname(os.path.abspath(__file__))
attendance_file = "fact_attendance.csv"
member_file = "dim_member.csv"
gender_file = "seeds\\gender.csv"
party_file = "seeds\\party.csv"
today_date = date.today().strftime("%Y-%m-%d")

### Main run here

attendance_df = pd.read_csv(os.path.join(script_dir, attendance_file))

gender_df = pd.read_csv(os.path.join(get_source_file_path(gender_file)))
party_df = pd.read_csv(os.path.join(get_source_file_path(party_file)))

df = summarise_attendance_information(attendance_df)

df = join_df(df, gender_df, "gender")
df = join_df(df, party_df, "party")

write_csv(os.path.join(script_dir, member_file), df)
