import requests
import json
import pandas as pd
import datetime
import pdb
from bs4 import BeautifulSoup

dates = pd.read_csv("seeds/dates.csv")
dates = dates.loc[dates['Date_Added'] == datetime.date.today().strftime('%Y-%m-%d')]

for index, row in dates.loc[dates['Version'] == 2].iterrows():
    date = row['Sitting_Date'] # 2012-09-10
    date_obj = datetime.datetime.strptime(date, '%Y-%m-%d')
    url = f"https://sprs.parl.gov.sg/search/getHansardReport/?sittingDate={date_obj.strftime('%d-%m-%Y')}"

    # Get Responses
    response = requests.get(url)

    print(date_obj)
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

    metadata = data["metadata"]
    sitting_cid = f"{metadata['parlimentNO']:03}-{metadata['sessionNO']:03}-{metadata['volumeNO']:03}-{metadata['sittingNO']:03}"
    sitting_date = datetime.datetime.strptime(metadata["sittingDate"], "%d-%m-%Y")

    sitting_df = pd.DataFrame({
        "Sitting_CID": [sitting_cid],
        "Sitting_Date": [sitting_date],
        "Parliament Session Str": [metadata["partSessionStr"]],
        "Parliament_Number": [metadata["parlimentNO"]],
        "Session_Number": [metadata["sessionNO"]],
        "Volume_Number": [metadata["volumeNO"]],
        "Sitting_Number": [metadata["sittingNO"]],
    })

    sitting_df.to_csv(f"code_output/{date_obj.strftime('%Y-%m-%d')}-sitting.csv", index=False, mode='w')

    # Get attendance information

    attendance_list = data["attendanceList"]
    member_names = [attendance["mpName"] for attendance in attendance_list]
    attendance_bool = [attendance["attendance"] for attendance in attendance_list]

    attendance_df = pd.DataFrame({
        "Date": [date_obj.strftime("%Y-%m-%d")] * len(attendance_list),
        "Sitting_CID": [sitting_cid] * len(attendance_list),
        "MP_Name": member_names,
        "Attendance": attendance_bool,
    })

    attendance_df.to_csv(f"code_output/{date_obj.strftime('%Y-%m-%d')}-attendance.csv", index=False, mode='w')

    # Get topic information

    takes_section_vo_list = data["takesSectionVOList"]

    titles = [section["title"] for section in takes_section_vo_list]
    subtitles = [section["subTitle"] for section in takes_section_vo_list]
    section_types = [section["sectionType"] for section in takes_section_vo_list]
    question_counts = [section["questionCount"] for section in takes_section_vo_list]

    order = list(range(1, len(takes_section_vo_list) + 1))

    topic_cid = [f"{sitting_cid}-{o:03}" for o in order]

    topics_df = pd.DataFrame({
        "Date": [date_obj.strftime("%Y-%m-%d")] * len(takes_section_vo_list),
        "Topic_CID": topic_cid,
        "Sitting_CID": [sitting_cid] * len(takes_section_vo_list),
        "Order": order,
        "Title": titles,
        "Subtitle": subtitles,
        "Section_Type": section_types,
        "Question_Count": question_counts,
    })

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

    speeches_df.to_csv(f"code_output/{date_obj.strftime('%Y-%m-%d')}-speeches.csv", index=False, mode='w', encoding='utf-8-sig')
