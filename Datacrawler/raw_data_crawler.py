# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.common.exceptions import NoSuchElementException, TimeoutException
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

import os
from langchain.utilities import GoogleSerperAPIWrapper
import pprint

class WebSearch:
    def __init__(self, api_key: str):

        self.api_key = api_key
        os.environ["SERPER_API_KEY"] = self.api_key
        self.search_wrapper = GoogleSerperAPIWrapper()

    #web search function
    def search(self, query: str) -> dict:

        return self.search_wrapper.results(query)

    def get_first_link(self, query: str) -> str:
        """
        获取搜索结果中的第一个链接。
        """
        search_result = self.search(query)
        if 'organic' in search_result and len(search_result['organic']) > 0:
            return search_result['organic'][0]['link']
        else:
            return "No link found"

    def webpilot_query(query):
        print(query)
        data = {
            "model": "wp-watt-3.52-16k",
            "content": query
        }
        headers = {
            'Authorization': f'Bearer 0a4e3011e3c3489c8b4ccb397b95017e'  # 注意：令牌不需要尖括号
        }
        response = requests.post('https://beta.webpilotai.com/api/v1/watt', json=data, headers=headers, stream=True)
        response_string = ""
        try:
            response_string = json.loads(response.text)["content"]
        except (KeyError, JSONDecodeError):
            response_string = "Webpilot API Error"
        except SSLError:
            print("SSL error! Retry in 5 minutes")
            time.sleep(300)
            webpilot_query(query)
        print(response_string)
        return response_string


class AutoPaginator:
    def __init__(self, start_url: str):
        """
        :param start_url: 分页的起始 URL
        """
        self.start_url = start_url
        self.url_list = []
        self.driver = webdriver.Chrome()

    def find_next_button(self):

        # 可能的选择器列表
        selectors = [
            {"by": By.XPATH, "value": "//a[contains(@aria-label, 'next')]"},  # 基于 `aria-label`
            {"by": By.XPATH, "value": "//a[contains(text(), 'Next')]"},  # 基于文本内容
            {"by": By.XPATH, "value": "//a[contains(text(), 'next') or contains(text(), '下一页')]"},  # 处理中英文
            {"by": By.CLASS_NAME, "value": "next"},  # 基于 `class`
            {"by": By.CSS_SELECTOR, "value": "a[rel='next']"},  # 基于 `rel` 属性
        ]

        for selector in selectors:
            try:
                # 尝试找到元素并返回
                element = WebDriverWait(self.driver, 5).until(
                    EC.element_to_be_clickable((selector["by"], selector["value"]))
                )
                return element
            except (NoSuchElementException, TimeoutException):
                continue  # 如果未找到元素，尝试下一个选择器

        return None

    def auto_paginate(self):
        """
        自动处理分页并将所有页码的 URL 存储在 url_list 中
        """
        self.driver.get(self.start_url)
        self.url_list.append(self.start_url)

        while True:
            # 解析当前页内容
            response = requests.get(self.driver.current_url)
            html = response.text
            soup = BeautifulSoup(html, 'html.parser')

            # 尝试查找“下一页”链接
            next_button = self.find_next_button()

            if next_button:
                next_button.click()
                self.url_list.append(self.driver.current_url)
            else:
                print("没有找到next button，分页结束。")
                break

        # 关闭浏览器
        self.driver.quit()

    def get_all_urls(self):
        return self.url_list


# 使用示例
if __name__ == "__main__":
    '''WebSearch搜url使用示例'''
    _query = 'faculty directory mechanical engineering Massachusetts Institute of Technology (MIT)'
    web_search = WebSearch(api_key="88a8892a02409063f02a3bb97ac08b36fb213ae7")
    first_link = web_search.get_first_link(query=_query)# 获取第一个link
    print(first_link)

    '''自动分页使用示例'''
    start_url = "https://engineering.msu.edu/faculty?departments=d09c48c1-b896-494b-8f18-fcbcbaa009e6&letter="
    paginator = AutoPaginator(start_url)
    paginator.auto_paginate()
    print("所有页码的 URL：", paginator.get_all_urls())