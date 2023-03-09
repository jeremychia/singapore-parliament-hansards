import requests
import json
from bs4 import BeautifulSoup
import pandas as pd
import datetime

date = '14-02-2023'
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

# Get title and content
titles = [takesSectionVO["title"] for takesSectionVO in data["takesSectionVOList"]]
contents = [takesSectionVO["content"] for takesSectionVO in data["takesSectionVOList"]]

df = pd.DataFrame(columns=['Date', 'Title', 'Speaker', 'Text', 'Seq'])

for i in range(len(contents)):
    # Parse through content
    soup = BeautifulSoup(contents[i], 'html.parser')

    speakers = []
    texts = []
    seq = []

    for p in soup.find_all('p'):
        if p.find("strong"):
            speakers.append(str(p.strong.text).strip())
            texts.append(str(p.find("strong").next_sibling).strip().replace('\xa0', ' ').replace(':', ' '))
            j = 1
            seq.append(j)
        elif not p.find("strong") and soup.find_all('p').index(p) != 0:
            speakers.append(speakers[soup.find_all('p').index(p) - 1])
            texts.append(str(p.text).strip().replace('\xa0', ' ').replace(':', ' '))
            j += 1
            seq.append(j)
        elif not p.find("strong") and soup.find_all('p').index(p) == 0:
            speakers.append('')
            texts.append(str(p.text).strip().replace('\xa0', ' ').replace(':', ' '))
            j = 1
            seq.append(j)

    # Create dataframe
    df_temp = pd.DataFrame({'Date': [date_obj.strftime('%Y-%m-%d')] * len(speakers),
                           'Title': [titles[i]] * len(speakers),
                           'Speaker': speakers,
                           'Text': texts,
                           'Seq': seq})

    df = pd.concat([df, df_temp], ignore_index = True)

df.to_csv(f"code_output/{date_obj.strftime('%Y-%m-%d')}.csv", index=False, mode='w')
