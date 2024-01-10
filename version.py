import pandas as pd
import datetime

dates = []  # get from royce's json

dates_fmt = [datetime.datetime.strptime(date, "%d-%m-%Y") for date in dates]

cutoff = datetime.datetime.strptime("10-09-2012", "%d-%m-%Y")

version_fmt = [1 if date < cutoff else 2 for date in dates_fmt]

version_df = pd.DataFrame({"Sitting_Date": dates_fmt, "Version": version_fmt})

version_df.to_csv(f"seeds/dates.csv", index=False, mode="w")
