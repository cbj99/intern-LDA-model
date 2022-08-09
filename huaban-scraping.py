from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.by import By
import pandas as pd
import time
import os.path

# 花瓣网网址
url = 'https://huaban.com/discovery'
# 登录使用的手机号或邮箱
user_id = ''
# 密码
password = ''
# 窗口下拉的次数
scroll_number = 30
# 用来存image信息的container
image_info_list = []

driver = webdriver.Chrome()
driver.get(url)
# 登录
driver.find_element(By.XPATH, '//*[@id="__next"]/main/div[1]/div/div/div[5]').click()
driver.implicitly_wait(5)
driver.find_element(By.XPATH, '//*[@id="email"]').send_keys(user_id)
driver.find_element(By.XPATH, '//*[@id="password"]').send_keys(password)
driver.find_element(By.XPATH, '/html/body/div[2]/div/div[2]/div/div[2]/div/div[2]/button').click()
driver.implicitly_wait(5)
# 从登陆后的页面再回到探索页面
time.sleep(1)
driver.find_element(By.XPATH, '/html/body/div[2]/div/div[2]/div/div[2]/div/div/div[2]/button').click()
time.sleep(1)
driver.find_element(By.XPATH, '//*[@id="__next"]/main/div[1]/div/div/div[1]/a[2]').click()

# 下拉窗口
time.sleep(3)
previous_height = driver.execute_script('return document.body.scrollHeight')
counter = 0

while counter<scroll_number:
    driver.execute_script('window.scrollTo(0, document.body.scrollHeight);')

    time.sleep(3)

    new_height = driver.execute_script('return document.body.scrollHeight')
    if new_height == previous_height:
        break
    previous_height = new_height
    counter+=1

    # 开始抓取
    images = driver.find_elements(By.CLASS_NAME, '__859QD3DN.zjSSb8O3')
    # print(images)
    for image in images:
        # print(image.get_attribute('innerHTML'))
        link = image.find_element(By.CLASS_NAME, 'flu_QVTL.hb-image').get_attribute('src')
        title = image.find_element(By.CLASS_NAME, 'flu_QVTL.hb-image').get_attribute('alt').replace('\r','').replace('\n','')
        if not title:
            title = '无标题'
        author = image.find_element(By.CLASS_NAME, 'nc3YzM55').text
        tag = image.find_element(By.CLASS_NAME, 'i_iYL3qa').text
        image_item = {
            'title': title,
            'author': author,
            'tag': tag,
            'link': link 
        }
        
        image_info_list.append(image_item)

    time.sleep(3)

# 存入csv
df = pd.DataFrame(image_info_list)

if(os.path.isfile('result.csv')):
    # 如果result.csv已存在
    df.to_csv('result.csv', encoding='utf-8', index=False, mode='a', header=False)
else:
    df.to_csv('result.csv', encoding='utf-8', index=False)

# 删除csv中的重复行
frame= pd.read_csv('result.csv', engine='python')
data = frame.drop_duplicates(keep='first')
data.to_csv('result.csv', encoding='utf-8', index=False)
driver.quit()