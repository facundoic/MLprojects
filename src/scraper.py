from bs4 import BeautifulSoup
from selenium import webdriver
import time

chromedriver = '/home/facundoic/Desktop/GitHub/ML-repository/MLprojects/chromedriver'
options = webdriver.ChromeOptions()
options.add_argument('--start-maximized')
options.add_argument('-ignore-certificate-errors')
options.add_argument('--incognito')

urls = [
    'DotCSV',
    'HolaMundoDev'
]
def driver():
    driver = webdriver.Chrome(chromedriver,chrome_options=options)
    for url in urls:
        driver.get('https://www.youtube.com/c/{}/videos?view=0&sort=p&flow=grid'.format(url))
        content = driver.page_source.encode('utf-8').strip()
        soup = BeautifulSoup(content,'html.parser')
        time.sleep(2)
        
        titles = soup.findAll('a',id='video-title')
        title_text = [title.text for title in titles[:10]]
        
        views = soup.findAll('span',class_='style-scope ytd-grid-video-renderer')
        video_antiquity = []
        view_text = [view.text for index,view in enumerate(views) if index % 2 == 0][:10]
        video_antiquity = [antiquity.text for index,antiquity in enumerate(views) if index % 2 != 0][:10]
    
        print(url+' most viewed videos')
        for i in range(0,10):
            print('Top {}: {} with {} from {}'.format(i+1,title_text[i],view_text[i],video_antiquity[i]))
        print('---------------------------------------------------------------------------------')
        time.sleep(2)
driver()
