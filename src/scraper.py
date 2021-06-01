from bs4 import BeautifulSoup
from selenium import webdriver
import time

chromedriver = '/home/facundoic/Desktop/GitHub/ML-repository/MLprojects/chromedriver'
options = webdriver.ChromeOptions()
options.add_argument('--start-maximized')
options.add_argument('-ignore-certificate-errors')
options.add_argument('--incognito')

urls = [
    'DotCSV'
]
def driver():
    driver = webdriver.Chrome(chromedriver,chrome_options=options)
    driver.get('https://www.youtube.com/c/{}/videos?view=0&sort=p&flow=grid'.format(urls[0]))
    content = driver.page_source.encode('utf-8').strip()
    soup = BeautifulSoup(content,'html.parser')
    time.sleep(2)
    """
    button_order = driver.find_element_by_xpath("//*[@id='trigger']").click()
    button_popular = driver.find_element_by_xpath("//*[@id='menu'']/a[1]").click()
    """
    titles = soup.findAll('a',id='video-title')
    title_text = [title.text for title in titles[:10]]
    
    views = soup.findAll('span',class_='style-scope ytd-grid-video-renderer')
    video_antiquity = []
    view_text = [view.text for index,view in enumerate(views) if index % 2 == 0][:10]
    video_antiquity = [antiquity.text for index,antiquity in enumerate(views) if index % 2 != 0][:10]
   
    print(urls[0]+'most viewed videos')
    for i in range(0,10):
        print('Top {}: {} with {} from {}'.format(i,title_text[i],view_text[i],video_antiquity[i]))

driver()
