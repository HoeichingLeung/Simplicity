#!/usr/bin/env python

import re
from time import sleep
from typing import Optional, Union, Generator, List, Any

from requests_html import HTMLSession
import json
import time
import random


class Spider:
    """Base class for all web spiders.

    Attributes:
        name: The name of the spider.
        start_urls: The list of URLs to start scraping from.
        extra_urls: The URLs that added after the start URLs.
        requests_delay: The delay between requests.
    """

    name: Optional[str] = None
    start_urls: Optional[List[str]] = []
    extra_urls: Optional[List[str]] = []
    requests_delay: Optional[Union[int, float]] = 0
    max_publications: Optional[int] = None  # Add this line to set the maximum number of publications

    def __init__(self) -> None:

        self.name = self.name if self.name is not None else self.__class__.__name__
        self.session = HTMLSession()
    def __str__(self) -> str:
        return f'{self.__class__.__name__}(name={repr(self.name)}, start_urls={self.start_urls}, extra_urls={self.extra_urls})'

    def __repr__(self) -> str:
        return str(self)

    def __eq__(self, other: object) -> bool:
        return self.start_urls == other.start_urls and self.extra_urls == other.extra_urls

    def set_max_publications(self, max_publications: int) -> None:
        """Sets the maximum number of publications to parse."""
        self.max_publications = max_publications
        self.max_publications_count = 0

    def setup(self, *args, **kwargs) -> None:
        """Setups the spiders before starting."""
        if kwargs.get('start_urls') is not None:
            self.start_urls = kwargs.get('start_urls')
        if kwargs.get('extra_urls') is not None:
            self.extra_urls = kwargs.get('extra_urls')

    def fetch(self) -> Generator[Any, None, None]:
        """Goes between all the specified URLs."""
        print(f'{self.name} started...')
        requests_count = 0
        for url in self.start_urls:
            requests_count += 1
            # print(f'\nFetching request #{requests_count} from {url}')
            response = self.session.get(url)
            for item in self.parse(response):
                yield item
            sleep(self.requests_delay)

        for url in self.extra_urls:
            requests_count += 1
            print(f'\nFetching request #{requests_count} from {url}')
            response = self.session.get(url)
            for item in self.parse(response):
                yield item
            sleep(self.requests_delay)

        print('Done!')

    def parse(self, response: HTMLSession) -> Generator[Any, None, None]:
        """Parses the response and yields items."""
        raise NotImplementedError


def _build_articles_query(**kwargs) -> str:
    """Builds the articles and case law query from the given arguments."""
    keywords = f'q={kwargs.get("keywords").replace(" ", "+")}'
    year_range = f'as_ylo={kwargs.get("start_year")}&as_yhi={kwargs.get("end_year")}'
    languages = f'lr={"|".join([f"lang_{l}" for l in kwargs.get("languages", [])])}'
    return f'hl=en&{keywords}&{year_range}&{languages}&{kwargs.get("extra", "")}'


class GSArticlesSpider(Spider):
    """Google Scholar articles category spider."""

    name = 'Google Scholar articles spider'
    requests_delay = .5

    def setup(self, *args, **kwargs) -> None:
        query = _build_articles_query(**kwargs, extra='as_sdt=0,5')
        self.start_urls = []
        self.start_urls.append(f'https://scholar.google.com/scholar?{query}')

    def parse(self, response: HTMLSession) -> Generator[dict, None, None]:

        import re
        results = []
        articles = response.html.xpath('//div[@class="gs_r gs_or gs_scl"]')
        for article in articles:
            print("publication_count = ", self.max_publications_count)

            time.sleep(random.uniform(1.1, 2.4))  # 休眠 2 到 5 秒之间的随机时间
            if self.max_publications is not None and self.max_publications_count >= self.max_publications:
                print('Reached the maximum number of publications to parse.')
                return  # Exit the function if the limit is reached

            # Extract article details with relative XPath
            title_elem = article.xpath('.//h3[@class="gs_rt"]/a | .//h3[@class="gs_rt"]/span[2]', first=True)
            title = title_elem.text if title_elem else None  # Ensure title is safely handled

            snippet_elem = article.xpath('.//div[@class="gs_a"]', first=True)
            snippet = snippet_elem.text if snippet_elem else ""
            print("snippet:",snippet)
            # Split snippet into parts and handle potential errors
            snippet_parts = [elem.strip() for elem in snippet.replace('\xa0', '').split('- ')]

            # Extract authors correctly from the first part of snippet
            authors = snippet_parts[0] if len(snippet_parts) > 0 else None

            # Correct the logic to handle potential variations in the snippet parts
            year = None
            if len(snippet_parts) >= 2:
                # Use regex to find year in possible locations in the snippet
                year_match = re.search(r'\d{4}', snippet_parts[-2]) or re.search(r'\d{4}', snippet_parts[-1])
                year = year_match.group() if year_match else None

            # Extract source and paper URL correctly
            source = article.xpath('.//h3[@class="gs_rt"]/a/@href', first=True)
            paper = article.xpath('.//div[@class="gs_or_ggsm"]/a/@href', first=True)

            # Extract citations number correctly
            citations_elem = article.xpath(
                './/div[@class="gs_ri"]/div[contains(@class, "gs_fl")]/a[contains(., "Cited by")]', first=True)
            citations_no = citations_elem.text.replace('Cited by ', '') if citations_elem is not None else None

            # Increment the publication counter
            self.max_publications_count += 1

            yield {
                'title': title,
                'authors': authors,
                'year': year,
                'source': source,
                'paper': paper,
                'citations no.': citations_no,
            }

            # # Write the results to a JSON file
            # with open('parsed_results_test.json', 'w', encoding='utf-8') as f:
            #     json.dump(results, f, ensure_ascii=False, indent=4)

        next_page = response.html.xpath('//div[@id="gs_res_ccl_bot"]//td[@align="left"]/a/@href', first=True)
        if next_page is not None:
            next_page_url = f'https://scholar.google.com{next_page}'
            print(f'Next page found at {next_page_url}')
            self.extra_urls = []
            self.extra_urls.append(next_page_url)


class GSCaseLawSpider(Spider):
    """Google Scholar case law category spider."""

    name = 'Google Scholar case law spider'
    requests_delay = .5

    def setup(self, *args, **kwargs) -> None:
        query = _build_articles_query(**kwargs, extra='as_sdt=2006')
        self.start_urls.append(f'https://scholar.google.com/scholar?{query}')

    def parse(self, response: HTMLSession) -> Generator[dict, None, None]:

        articles = response.html.xpath('//div[@class="gs_r gs_or gs_scl"]')
        for article in articles:
            if self.max_publications is not None and self.max_publications_count >= self.max_publications:
                print('Reached the maximum number of publications to parse.')
                return  # Exit the function if the limit is reached

            # Extract article details
            title = article.xpath('//h3[@class="gs_rt"]/a | .//h3[@class="gs_rt"]/span[2]', first=True).text
            snippet = article.xpath('//div[@class="gs_a"]', first=True).text
            snippet = [elem.strip() for elem in snippet.replace('\xa0', '').split('- ')]
            authors = snippet[0]
            if year := None or len(snippet) >= 2:
                year = re.search(r'\d{4,}$', snippet[-2]) or re.search(r'\d{4,}$', snippet[-1])
                year = year.group() if year is not None else None
            source = article.xpath('//h3[@class="gs_rt"]/a/@href', first=True)
            paper = article.xpath('//div[@class="gs_or_ggsm"]/a/@href', first=True)
            citations_no = article.xpath(
                '//div[@class="gs_ri"]/div[contains(@class, "gs_fl")]/a[3][contains(., "Cited by")]', first=True)
            citations_no = citations_no.text.replace('Cited by ', '') if citations_no is not None else None

            self.max_publications_count += 1  # Increment the counter
            yield {
                'title': title,
                'authors': authors,
                'year': year,
                'source': source,
                'paper': paper,
                'citations no.': citations_no,
            }

        next_page = response.html.xpath('//div[@id="gs_res_ccl_bot"]//td[@align="left"]/a/@href', first=True)
        if next_page is not None:
            next_page_url = f'https://scholar.google.com{next_page}'
            print(f'Next page found at {next_page_url}')
            self.extra_urls.append(next_page_url)


class GSProfilesSpider(Spider):
    """Google Scholar profiles category spider."""

    name = 'Google Scholar profiles spider'
    requests_delay = .5

    def setup(self, *args, **kwargs) -> None:
        query = f'hl=en&user={kwargs.get("keywords")}&cstart=0&pagesize=100'
        self.start_urls.append(f'https://scholar.google.com/citations?{query}')
        self.cstart = 0

    def parse(self, response: HTMLSession) -> Generator[dict, None, None]:

        articles = response.html.xpath('//div[@class="gs_r gs_or gs_scl"]')
        for article in articles:
            if self.max_publications is not None and self.max_publications_count >= self.max_publications:
                print('Reached the maximum number of publications to parse.')
                return  # Exit the function if the limit is reached

            # Extract article details
            title = article.xpath('//h3[@class="gs_rt"]/a | .//h3[@class="gs_rt"]/span[2]', first=True).text
            snippet = article.xpath('//div[@class="gs_a"]', first=True).text
            snippet = [elem.strip() for elem in snippet.replace('\xa0', '').split('- ')]
            authors = snippet[0]
            if year := None or len(snippet) >= 2:
                year = re.search(r'\d{4,}$', snippet[-2]) or re.search(r'\d{4,}$', snippet[-1])
                year = year.group() if year is not None else None
            source = article.xpath('//h3[@class="gs_rt"]/a/@href', first=True)
            paper = article.xpath('//div[@class="gs_or_ggsm"]/a/@href', first=True)
            citations_no = article.xpath(
                '//div[@class="gs_ri"]/div[contains(@class, "gs_fl")]/a[3][contains(., "Cited by")]', first=True)
            citations_no = citations_no.text.replace('Cited by ', '') if citations_no is not None else None

            self.max_publications_count += 1  # Increment the counter
            yield {
                'title': title,
                'authors': authors,
                'year': year,
                'source': source,
                'paper': paper,
                'citations no.': citations_no,
            }

        next_page = response.html.xpath('//div[@id="gs_res_ccl_bot"]//td[@align="left"]/a/@href', first=True)
        if next_page is not None:
            next_page_url = f'https://scholar.google.com{next_page}'
            print(f'Next page found at {next_page_url}')
            self.extra_urls.append(next_page_url)


def process_publications(_dataframe):
    df = _dataframe
    for index, row in df.iterrows():
        prof_name = row['Faculty']
        university_name = row['University']

        # Define parameters for the search
        params = {
            'keywords': f'{prof_name} {university_name}',  # Change keywords as needed
            'start_year': 2014,              # Start year for the search
            'end_year': 2024,                # End year for the search
            'languages': ['en'],             # Language(s) for the search
        }

        # Choose the spider based on what you want to search
        gs_spider = GSArticlesSpider()  # Use GSCaseLawSpider() or GSProfilesSpider() for different searches

        # Set up the spider with the given parameters
        gs_spider.setup(**params)
        gs_spider.set_max_publications(10)  # Set the maximum number of publications to parse

        # Fetch the results
        results = pd.DataFrame(gs_spider.fetch())

        # Check and save results
        if results.empty:
            print('No results found. Try again later or change your IP address.')
        else:
            results.sort_values('citations no.', ascending=False, inplace=True)

        '''
        results转成str写到csv最后一列publication
        '''

        if results.empty:
            print('No results found. Try again later or change your IP address.')
        else:
            results.sort_values('citations no.', ascending=False, inplace=True)

            # string into column 'publications'
            result_str = results.apply(lambda x: f"Title: {x['title']}, Authors: {x['authors']}, Year: {x['year']}, Source: {x['source']}", axis=1).str.cat(sep=' | ')
            df.at[index, 'publications'] = result_str
    return df


if __name__ == '__main__':
    import os
    import sys
    import argparse
    from pathlib import Path
    import pandas as pd
    import chardet
    original_file_path = 'D:\llm_python\Simplicity\data\CS_AI.csv'  # 替换为你的CSV文件路径
    # 检测文件编码
    with open(original_file_path, 'rb') as f:
        result = chardet.detect(f.read())

    print(f"Detected encoding: {result['encoding']}")  # 输出检测到的编码

    # 读取原始 CSV 文件
    df = pd.read_csv(original_file_path, encoding=result['encoding'])

    # 定义每个子文件的行数
    chunk_size = 20

    ## 遍历数据帧并分割成多个文件
    # for i in range(0, len(df), chunk_size):
    #     chunk = df.iloc[i:i + chunk_size]
    #     chunk.to_csv(f'D:/llm_python/Simplicity/data/txt_to_embed/split_pub_cs_ai/part_{i // chunk_size}.csv', index=False)


    # 列出所有分割的 CSV 文件路径
    directory_path = 'D:/llm_python/Simplicity/data/txt_to_embed/split_pub_cs_ai/'
    files = [os.path.join(directory_path, file) for file in os.listdir(directory_path) if file.startswith('part_')]

    for file_path in files:
        print("file_path_current:", file_path)
        df = pd.read_csv(file_path)
        updated_df = process_publications(df)
        # 保存更新后的数据到新的 CSV 文件
        updated_df.to_csv(file_path.replace('part_', 'updated_part_'), index=False)

    updated_files = [os.path.join(directory_path, file) for file in os.listdir(directory_path) if
                     file.startswith('updated_part_')]
    # 合并所有更新的 CSV 文件

    df_list = [pd.read_csv(file) for file in updated_files]
    print(df_list)
    final_df = pd.concat(df_list, ignore_index=True)

    # 保存合并后的最终 CSV 文件
    final_df.to_csv('D:/llm_python/Simplicity/data/Updated_CS_AI_completed.csv',
                    index=False)


    #
    # df = pd.read_csv(file_path, encoding=result['encoding'])
    # df['publications'] = ''
    # for index, row in df.iterrows():
    #     prof_name = row['Faculty']
    #     university_name = row['University']
    #
    #     # Define parameters for the search
    #     params = {
    #         'keywords': f'{prof_name}_{university_name}',  # Change keywords as needed
    #         'start_year': 2018,              # Start year for the search
    #         'end_year': 2024,                # End year for the search
    #         'languages': ['en'],             # Language(s) for the search
    #     }
    #
    #     # Choose the spider based on what you want to search
    #     gs_spider = GSArticlesSpider()  # Use GSCaseLawSpider() or GSProfilesSpider() for different searches
    #
    #     # Set up the spider with the given parameters
    #     gs_spider.setup(**params)
    #     gs_spider.set_max_publications(10)  # Set the maximum number of publications to parse
    #
    #     # Fetch the results
    #     results = pd.DataFrame(gs_spider.fetch())
    #
    #     # Check and save results
    #     if results.empty:
    #         print('No results found. Try again later or change your IP address.')
    #     else:
    #         results.sort_values('citations no.', ascending=False, inplace=True)
    #
    #     '''
    #     results转成str写到csv最后一列publication
    #     '''
    #
    #     if results.empty:
    #         print('No results found. Try again later or change your IP address.')
    #     else:
    #         results.sort_values('citations no.', ascending=False, inplace=True)
    #
    #         # string into column 'publications'
    #         result_str = results.apply(lambda x: f"Title: {x['title']}, Authors: {x['authors']}, Year: {x['year']}, Source: {x['source']}", axis=1).str.cat(sep=' | ')
    #         df.at[index, 'publications'] = result_str
    #
    # # 保存更新后的DataFrame到CSV
    # output_file_path = 'D:\\llm_python\\Simplicity\\data\\Updated_Physics_Semiconductor_or_Quantum_Computation_or_Quantum_Optics_completed.csv'
    # df.to_csv(output_file_path, index=False)
    # print(f'Results saved to {output_file_path}')