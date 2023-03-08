import time
from selenium import webdriver
from selenium.common.exceptions import StaleElementReferenceException
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager

# Start maximised
options = Options()
options.add_argument("start-maximized")

# Set up the Chrome web driver
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

url = 'https://sprs.parl.gov.sg/search/#/sprs3topic?reportid=budget-2054' #2034
driver.get(url)

# Wait for the page to fully load
driver.implicitly_wait(10)

# Find all paragraphs and concatenate the text contents
speeches = []


# Find all the p elements on the page
paragraphs = driver.find_elements(By.TAG_NAME, 'p')
print("Number of Paragraphs: "+ str(len(paragraphs)))

# Iterate over the paragraphs and get their text
for p in paragraphs:
    try:
        p_text = p.text
        print("Iteration (Standard): " + str(paragraphs.index(p)))
        print(p_text)
    except StaleElementReferenceException:
        alternate_element = driver.find_element(By.XPATH, "//p[" + str(paragraphs.index(p)+1) + "]" )
        p_text = alternate_element.text
        print("Iteration (Exception): " + str(paragraphs.index(p)))
        print(p_text)
    finally:
        speeches.append(p_text)
        # Do something with the paragraph text
        

print(speeches)

# Close the web driver
driver.quit()
