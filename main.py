import requests
import json

url = "https://sprs.parl.gov.sg/search/getHansardReport/?sittingDate=14-02-2023"

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