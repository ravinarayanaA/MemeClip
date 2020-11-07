import time
import urllib.request
import collections
import pandas as pd
from bs4 import BeautifulSoup
from selenium import webdriver

browser = webdriver.Chrome('C:\\Work\\aardvarc\\drivers\\chromedriver.exe')
browser.get("https://memebot.life/")
time.sleep(7)
text = browser.page_source
soup = BeautifulSoup(text)
data = soup.findAll('button',attrs={'class':'meme'})
opener = urllib.request.build_opener()
opener.addheaders = [('User-agent', 'Mozilla/5.0')]
urllib.request.install_opener(opener)
audio = collections.defaultdict(list)

for div in data:
    name = div.find("p",attrs={"class":"name"}).get_text()
    url = "https://memebot.nyc3.cdn.digitaloceanspaces.com/memebot/audio/"+name.strip()+".opus"
    file_name = "C:/Class Documents/DL/project/audio_data_downloader/memlife/"+name.strip()+".opus"
    urllib.request.urlretrieve(url, file_name)
    aliases_div = div.find("p",attrs={"class":"aliases"})
    aliases = ""
    if aliases_div:
        aliases = aliases_div.get_text().replace("aliases:","").strip()
    tags = div.find("ul",attrs={"class":"tags"})
    tags_list = []
    if tags:
        for tag in tags.children:
            tags_list.append(tag.get_text())
    audio["name"].append(name)
    audio["url"].append(url)
    audio["path"].append(file_name)
    audio["aliases"].append(aliases)
    audio["tags"].append(",".join(tags_list))
    print(name, aliases, tags_list)

df = pd.DataFrame(audio, columns=['name', 'url', 'path', 'aliases', 'tags'])
df.to_csv(r'C:/Class Documents/DL/project/audio_data_downloader/data_output.csv', index = False, header=True)
browser.close()
