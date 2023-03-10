import requests
import json
from bs4 import BeautifulSoup
import pandas as pd
import datetime

date = '27-02-2023'
date_obj = datetime.datetime.strptime(date, '%d-%m-%Y')
url = f"https://sprs.parl.gov.sg/search/getHansardReport/?sittingDate={date}"

# Get Responses
response = requests.get(url)

if response.status_code == 200:
    # Do something with the response data
    data = response.json()
    print("Request succeeded.")
elif response.status_code == 404:
    print("The requested resource was not found.")
elif response.status_code == 500:
    print("An internal server error occurred.")
else:
    print(f"Unexpected status code: {response.status_code}")

# Get sitting information

parliament_number = data["metadata"]["parlimentNO"]
session_number = data["metadata"]["sessionNO"]
volume_number = data["metadata"]["volumeNO"]
sitting_number = data["metadata"]["sittingNO"]
sitting_date = datetime.datetime.strptime(data["metadata"]["sittingDate"], '%d-%m-%Y')
parl_session_str = data["metadata"]["partSessionStr"]

session_cid = f'{parliament_number:03}-{session_number:03}-{volume_number:03}-{sitting_number:03}'

# Get title and content
titles = [takesSectionVO["title"] for takesSectionVO in data["takesSectionVOList"]]
contents = [takesSectionVO["content"] for takesSectionVO in data["takesSectionVOList"]]

speeches_df = pd.DataFrame(columns=['Date', 'Title', 'Speaker', 'Text', 'Seq'])

for i in range(len(contents)):
    # Parse through content
    soup = BeautifulSoup(contents[i], 'html.parser')

    speakers = []
    texts = []
    sequences = []

    for index, p in enumerate(soup.find_all('p')):
        if p.strong:
            speaker = str(p.strong.text).strip()
            text = str(p.find("strong").next_sibling)
            sequence = 1
        else:
            speaker = speakers[-1] if index > 0 else ''
            text = str(p.text)
            sequence = sequences[-1] + 1 if index > 0 else 1
        
        speakers.append(speaker)
        texts.append(text.strip().replace('\xa0', ' ').replace(':', ' '))
        sequences.append(sequence)

    # Create dataframe
    df_temp = pd.DataFrame({'Date': [date_obj.strftime('%Y-%m-%d')] * len(speakers),
                           'Title': [titles[i]] * len(speakers),
                           'Speaker': speakers,
                           'Text': texts,
                           'Seq': sequences})

    speeches_df = pd.concat([speeches_df, df_temp], ignore_index = True)

speeches_df.to_csv(f"code_output/{date_obj.strftime('%Y-%m-%d')}.csv", index=False, mode='w')
