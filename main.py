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

# Get sitting information

parliament_number = data["metadata"]["parlimentNO"]
session_number = data["metadata"]["sessionNO"]
volume_number = data["metadata"]["volumeNO"]
sitting_number = data["metadata"]["sittingNO"]
sitting_date = datetime.datetime.strptime(data["metadata"]["sittingDate"], '%d-%m-%Y')
parl_session_str = data["metadata"]["partSessionStr"]

sitting_cid = f'{parliament_number:03}-{session_number:03}-{volume_number:03}-{sitting_number:03}'

# Get attendance information

member_names = [attendance["mpName"] for attendance in data["attendanceList"]]
attendance_bool = [attendance["attendance"] for attendance in data["attendanceList"]]

attendance_df = pd.DataFrame({'Date': [date_obj.strftime('%Y-%m-%d')] * len(data["attendanceList"]),
                              'Sitting_CID': [sitting_cid] * len(data["attendanceList"]),
                              'MP_Name': member_names,
                              'Attendance': attendance_bool})

attendance_df.to_csv(f"code_output/{date_obj.strftime('%Y-%m-%d')}-attendance.csv", index=False, mode='w')

# Get topic information

titles = [takesSectionVO["title"] for takesSectionVO in data["takesSectionVOList"]]
subtitles = [takesSectionVO["subTitle"] for takesSectionVO in data["takesSectionVOList"]]
section_types = [takesSectionVO["sectionType"] for takesSectionVO in data["takesSectionVOList"]]
question_counts = [takesSectionVO["questionCount"] for takesSectionVO in data["takesSectionVOList"]]

order = list(range(1, len(data["takesSectionVOList"])+1))

topic_cid = [f'{sitting_cid}-{o:03}' for o in order]

topics_df = pd.DataFrame({'Date': [date_obj.strftime('%Y-%m-%d')] * len(data["takesSectionVOList"]),
                          'Topic_CID': topic_cid,
                          'Sitting_CID': [sitting_cid] * len(data["takesSectionVOList"]),
                          'Order': order,
                          'Title': titles,
                          'Subtitle': subtitles,
                          'Section_Type': section_types,
                          'Question_Count': question_counts})

topics_df.to_csv(f"code_output/{date_obj.strftime('%Y-%m-%d')}-topics.csv", index=False, mode='w')

# Get content

contents = [takesSectionVO["content"] for takesSectionVO in data["takesSectionVOList"]]

speeches_df = pd.DataFrame(columns=['Date', 'Title', 'Topic_CID', 'Speaker', 'Text', 'Seq', 'Topic_Seq', 'Speeches_CID'])

for i in range(len(contents)):
    # Parse through content
    soup = BeautifulSoup(contents[i], 'html.parser')

    speakers = []
    texts = []
    sequences = []
    cur_topic_cid = topic_cid[i]

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
                           'Topic_CID': [cur_topic_cid] * len(speakers),
                           'Title': [titles[i]] * len(speakers),
                           'Speaker': speakers,
                           'Text': texts,
                           'Seq': sequences,
                           'Topic_Seq': list(range(1, len(speakers)+1)),
                           'Speeches_CID': [f'{cur_topic_cid}-{o:03}' for o in list(range(1, len(speakers)+1))]})

    speeches_df = pd.concat([speeches_df, df_temp], ignore_index = True)

speeches_df['Sitting_Seq'] = list(range(1, len(speeches_df)+1))

speeches_df.to_csv(f"code_output/{date_obj.strftime('%Y-%m-%d')}-speeches.csv", index=False, mode='w')
