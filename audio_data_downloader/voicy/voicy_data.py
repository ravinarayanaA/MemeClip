import time
import urllib.request
import collections
import pandas as pd
from selenium import webdriver

browser = webdriver.Chrome('C:\\Work\\aardvarc\\drivers\\chromedriver.exe')
browser.get("https://www.voicy.network/")
time.sleep(3)
browser.find_element_by_class_name("pull-right").find_elements_by_xpath(".//*")[2].click()
time.sleep(3)
browser.find_element_by_id("UsernameLogin").send_keys("rnadkathimar@gmail.com")
browser.find_element_by_id("PasswordLogin").send_keys("Ravi@5250")
browser.find_elements_by_xpath('//button[@type="submit"]')[3].click()
time.sleep(3)
n = 50
for i in range(n):
    browser.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    time.sleep(2)
browser.execute_script("scroll(0, 0);")
clips = browser.find_elements_by_class_name("clip")
opener = urllib.request.build_opener()
opener.addheaders = [('User-agent', 'Mozilla/5.0')]
urllib.request.install_opener(opener)
audio = collections.defaultdict(list)
urls = {}
for clip in clips:
    page_url = clip.find_element_by_tag_name("a").get_attribute("href")
    mp3_url = clip.find_element_by_class_name("share-download").get_attribute("href")
    urls[page_url] = mp3_url

browser.close()
browser2 = webdriver.Chrome('C:\\Work\\aardvarc\\drivers\\chromedriver.exe')

for page in urls:
    browser2.get(page)
    time.sleep(4)
    title = browser2.find_element_by_class_name("clip-description").find_element_by_tag_name("a").text
    tags_elements = browser2.find_element_by_class_name("tags-detail").find_elements_by_tag_name("a")
    tags = []
    for tag in tags_elements:
        tags.append(tag.text[1:])
    audio["name"].append(title)
    audio["tags"].append(",".join(tags))
    audio["audio_url"].append(urls[page])
    audio["page_url"].append(page)
    file_name = "C:/Class Documents/DL/project/audio_data_downloader/voicy/dataset/" + urls[page].split("/")[-1]
    urllib.request.urlretrieve(urls[page], file_name)
    audio["file_path"].append(file_name)
browser2.close()
df = pd.DataFrame(audio, columns=['name', 'tags', 'audio_url', 'page_url', 'file_path'])
df.to_csv(r'C:/Class Documents/DL/project/audio_data_downloader/voicy/data_output.csv', index = False, header=True)



