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
import json
import time  
from requests.exceptions import SSLError  
from json.decoder import JSONDecodeError  
import csv  
import re 
import pandas as pd


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

    def webpilot_query(self, query):
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
            return webpilot_query(query)  

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

def get_basic_data_with_link():
    '''
    用于需要手动额外选择的学校
    '''
    web_search = WebSearch(api_key="88a8892a02409063f02a3bb97ac08b36fb213ae7")
    universities = [    
            #{"name": "RWTH Aachen University", "link": "faculty directory mechanical engineering of RWTH Aachen University", "rank": 19},  
            {"name": "Georgia Institute of Technology", "link": "https://www.me.gatech.edu/faculty?field_last_name_value=&field_staff_group_target_id=5&field_all_research_areas_target_id=All&page=0", "rank": 14}
            ]  

    results = []  

    for university in universities:    

        start_url = university['link'] 
        paginator = AutoPaginator(start_url)  
        paginator.auto_paginate()  
        url_list = paginator.get_all_urls()  
        print(f"All page URLs for {university['name']}: {url_list}")  

        result = []  
        for url in url_list:  
            query_head = """
            Please list the information of current professors from this URL in the format 
            '1. Name # Title # Research # Website # Email 2. Name # Title # Research # Website # Email',
            Research shows the research interest of the professors, website is the homepage,
            if you can not find some of the items like Research, Email or website, just use 'None'
            reducing unnecessary introductions and conclusions:"""  
            query = query_head + url  
            prof_name = web_search.webpilot_query(query)  
            result.append(prof_name)  

        combined_result = "\n".join(result) 
        output_txt_path = f'D:/Simplicity/data/faculty/{university["name"]}_result.txt'  

        # 确保目标目录存在  
        import os  
        os.makedirs(os.path.dirname(output_txt_path), exist_ok=True)  

        # 将 combined_result 保存到 .txt 文件中  
        with open(output_txt_path, 'w', encoding='utf-8') as txt_file:  
            txt_file.write(combined_result)  

        print(f"TXT 文件已保存：{output_txt_path}")
        

def get_detailed(csv_file_path):  
    '''
    获取Research Email Website等额外信息
    '''
    web_search = WebSearch(api_key="88a8892a02409063f02a3bb97ac08b36fb213ae7")
    # 读取CSV文件  
    df = pd.read_csv(csv_file_path)  
    
    # 用于存储所有教授信息的列表  
    all_prof_info = []  
    
    # 遍历每一行  
    for index, row in df.iterrows():  
        university = row['University']  
        faculty = row['Faculty']  
        
        # 构建查询字符串  
        query = f"{university} {faculty} homepage"  
        
        # 获取第一个链接作为主页  
        homepage = web_search.get_first_link(query=query)  
        
        # 构建查询请求  
        query_head = """  
            Please list the information of current professors from this URL in the format:  
            '1. Name # Title # Research # Website # Email 2. Name # Title # Research # Website # Email'.  
            For each professor, use the URL I provide as the Website field. If you cannot find some items like Research, Email, just use 'None'.  
            Please avoid unnecessary introductions and conclusions:  
            """  
        
        query = query_head + homepage  
        
        # 获取教授信息  
        prof_info = web_search.webpilot_query(query)  
        
        # 将教授信息添加到列表中  
        all_prof_info.append(prof_info)  
    
    # 将所有教授信息写入一个TXT文件  
    txt_file_path = f"{csv_file_path.split('.')[0]}.txt"  
    with open(txt_file_path, 'w', encoding='utf-8') as txt_file:  
        for info in all_prof_info:  
            txt_file.write(info + "\n")  
    
    print(f"Information saved to {txt_file_path}")  



def get_basic_data():
    #自动爬取测试
    web_search = WebSearch(api_key="88a8892a02409063f02a3bb97ac08b36fb213ae7")
    # 这一部分可以变成json读入
    universities = [  
            # ok{"name": "Delft University of Technology", "query": "faculty directory mechanical engineering of Delft University of Technology", "rank": 3},  
            #{"name": "University of Cambridge", "query": "faculty directory mechanical engineering of University of Cambridge", "rank": 4},  
            #{"name": "Harvard University", "query": "faculty directory mechanical engineering of Harvard University", "rank": 5},  
            # ok{"name": "ETH Zurich - Swiss Federal Institute of Technology", "query": "faculty directory mechanical engineering of ETH Zurich", "rank": 6},  
            # ok{"name": "National University of Singapore (NUS)", "query": "faculty directory mechanical engineering of National University of Singapore", "rank": 7},  
            #{"name": "University of California, Berkeley (UCB)", "query": "faculty directory mechanical engineering of University of California, Berkeley", "rank": 7},  
            #{"name": "Politecnico di Milano", "query": "faculty directory mechanical engineering of Politecnico di Milano", "rank": 9},  
            # ok{"name": "University of Oxford", "query": "faculty directory mechanical engineering of University of Oxford", "rank": 9},  
            # ok{"name": "Imperial College London", "query": "faculty directory mechanical engineering of Imperial College London", "rank": 11},  
            # ok{"name": "Nanyang Technological University, Singapore (NTU)", "query": "faculty directory mechanical engineering of Nanyang Technological University", "rank": 12}, 
              
            # ok{"name": "Georgia Institute of Technology", "query": "faculty directory mechanical engineering of Georgia Institute of Technology", "rank": 14},  
            # ok{"name": "University of Michigan-Ann Arbor", "query": "faculty directory mechanical engineering of University of Michigan-Ann Arbor", "rank": 15},  
            # ok{"name": "Purdue University", "query": "faculty directory mechanical engineering of Purdue University", "rank": 16},
            # ok{"name": "California Institute of Technology (Caltech)", "query": "faculty directory mechanical engineering of California Institute of Technology", "rank": 17} 
            # ok{"name": "EPFL", "query": "faculty directory mechanical engineering of EPFL", "rank": 18},  
            #{"name": "RWTH Aachen University", "query": "faculty directory mechanical engineering of RWTH Aachen University", "rank": 19},  
            # ok{"name": "KTH Royal Institute of Technology", "query": "faculty directory mechanical engineering of KTH Royal Institute of Technology", "rank": 20}
            ]  

    results = []  

    for university in universities:  
        '''
        _query = university["query"]  
        print(f"Searching for: {_query}")  
        first_link = web_search.get_first_link(query=_query)  
        print(f"First link for {university['name']}: {first_link}")  

        start_url = first_link  
        paginator = AutoPaginator(start_url)  
        paginator.auto_paginate()  
        url_list = paginator.get_all_urls()  
        print(f"All page URLs for {university['name']}: {url_list}")  

        result = []  
        for url in url_list:  
            query_head = """
            Please list the information of current professors from this URL in the format 
            '1. Name # Title # Research # Website # Email 2. Name # Title # Research # Website # Email',
            Research shows the research interest of the professors, website is the homepage,
            if you can not find some of the items like Research, Email or website, just use 'None'
            reducing unnecessary introductions and conclusions:"""  
            query = query_head + url  
            prof_name = web_search.webpilot_query(query)  
            result.append(prof_name)  

        combined_result = "\n".join(result)  
        '''
        '''
        # 如果有问题可以运行这一段代码作为检查
        # 定义文件保存路径  
        output_txt_path = f'D:/Simplicity/data/faculty/{university["name"]}_result.txt'  

        # 确保目标目录存在  
        import os  
        os.makedirs(os.path.dirname(output_txt_path), exist_ok=True)  

        # 将 combined_result 保存到 .txt 文件中  
        with open(output_txt_path, 'w', encoding='utf-8') as txt_file:  
            txt_file.write(combined_result)  

        print(f"TXT 文件已保存：{output_txt_path}")
        '''
        # 定义文件路径  
        file_path = f'D:/Simplicity/data/faculty/{university["name"]}_result.txt'  

        # 读取文件内容  
        with open(file_path, 'r', encoding='utf-8') as file:  
            combined_result = file.read()  
        #print(combined_result) 
        
        # Adjust the pattern to capture only five fields with '#' as the separator  
        # pattern = r"\d+\.\s+(.*?)\s*#\s*(.*?)\s*#\s*(.*?)\s*#\s*(.*?)\s*#\s*([\w\.-]+@[\w\.-]+)"  # 有邮箱
        pattern = r"\d+\.\s+(.*?)\s*#\s*(.*?)\s*#\s*(.*?)\s*#\s*(.*?)\s*#\s*(.*?)" # 没邮箱

        # Use re.findall to capture all groups  
        matches = re.findall(pattern, combined_result)  

        for name, title, research, website, email in matches:  
            results.append({  
                'Rank': university["rank"],  
                'University': university["name"],  
                'Faculty': name.strip(),  
                'Title': title.strip(),  
                'Research': research.strip() if research.strip() != 'None' else '',  
                'Website': website.strip() if website.strip() != 'None' else '',  
                'Email': email.strip() if email.strip() != 'None' else '',  
                'Status': '',  
                'is_chinese': '',  
                'send_priority': '',  
                'full_research': '',  
                'publications': ''  
            })

    # 确保目录存在  
    output_dir = 'D:/Simplicity/data/faculty'  
    os.makedirs(output_dir, exist_ok=True)  

    # 创建 CSV 文件  
    csv_file_path = os.path.join(output_dir, '18mechanical_engineering_faculty.csv')  
    with open(csv_file_path, 'w', newline='', encoding='utf-8') as csvfile:  
        fieldnames = ['Rank', 'University', 'Faculty', 'Title', 'Research', 'Website', 'Email', 'Status', 'is_chinese', 'send_priority', 'full_research', 'publications']  
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)  

        writer.writeheader()  
        for result in results:  
            writer.writerow(result)  

    print(f"CSV 文件已创建：{csv_file_path}")  


# 使用示例
if __name__ == "__main__":
    '''
    #WebSearch搜url使用示例
    _query = 'faculty directory mechanical engineering Massachusetts Institute of Technology (MIT)'
    web_search = WebSearch(api_key="88a8892a02409063f02a3bb97ac08b36fb213ae7")
    first_link = web_search.get_first_link(query=_query)# 获取第一个link
    print(first_link)

    #自动分页使用示例
    start_url = "https://engineering.msu.edu/faculty?departments=d09c48c1-b896-494b-8f18-fcbcbaa009e6&letter="
    paginator = AutoPaginator(start_url)
    paginator.auto_paginate()
    print("所有页码的 URL：", paginator.get_all_urls())
    '''
    #get_basic_data()
    get_detailed('D:/Simplicity/data/faculty/test.csv')
    #get_basic_data_with_link()
    