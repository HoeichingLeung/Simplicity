from dataclasses import dataclass, field
from typing import Optional, List, Tuple, List, Any
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import argparse
from datetime import datetime
import logging
from openai import OpenAI
import os

from queue import PriorityQueue
import numpy as np
import base64
from accessibility_tree import fetch_browser_info, fetch_page_accessibility_tree,\
                    parse_accessibility_tree, clean_accesibility_tree

def encode_image(image_path: str) -> str:
    """Encode image to base64 string"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

@dataclass
class PageNode:
    """表示搜索树中的一个网页节点"""
    url: str
    parent: Optional['PageNode'] = None
    children: List['PageNode'] = field(default_factory=list)
    depth: int = 0
    
    # 页面内容相关
    screenshot_path: str = ""
    accessibility_tree: str = ""
    extracted_text: str = ""
    available_links: List[str] = field(default_factory=list)
    
    # 状态标志
    is_visited: bool = False
    has_useful_info: bool = False
    relevance_score: float = 0.0
    
    # 元数据
    visit_time: Optional[datetime] = None
    processing_time: float = 0.0

class SearchTree:
    """管理搜索树结构"""
    def __init__(self, root_url: str, max_depth: int = 3):
        self.root = PageNode(url=root_url)
        self.current_node = self.root
        self.max_depth = max_depth
        self.visited_urls = set()
        self.useful_pages = []
        self.link_queue = PriorityQueue()
    
    def add_child(self, parent: PageNode, url: str) -> PageNode:
        """添加子节点"""
        child = PageNode(url=url, parent=parent, depth=parent.depth + 1)
        parent.children.append(child)
        return child
    
    def backtrack(self) -> Optional[PageNode]:
        """回溯到上一个有未访问链接的节点"""
        while self.current_node.parent is not None:
            self.current_node = self.current_node.parent
            if self.has_unvisited_links(self.current_node):
                return self.current_node
        return None
    
    def has_unvisited_links(self, node: PageNode) -> bool:
        """检查节点是否还有未访问的链接"""
        return any(url not in self.visited_urls for url in node.available_links)



class WebAgent:
    def __init__(self, 
                 llm_client: Any,
                 output_dir: str = 'results',
                 relevance_threshold: float = 0.7,
                 max_depth: int = 3,
                 max_pages: int = 50,
                 time_limit: int = 3600,
                 window_width: int = 1024,
                 window_height: int = 768):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.browser = self._setup_browser(window_width, window_height)
        self.llm_client = llm_client
        self.search_tree = None
        self.relevance_threshold = relevance_threshold
        self.max_depth = max_depth
        self.max_pages = max_pages
        self.time_limit = time_limit
        self.start_time = None
        
    def _setup_browser(self, window_width: int, window_height: int) -> webdriver.Chrome:
        """设置浏览器"""
        chrome_options = Options()
        chrome_options.add_argument('--headless')
        chrome_options.add_argument(f'--window-size={window_width},{window_height}')
        chrome_options.add_argument('--disable-gpu')
        browser = webdriver.Chrome(options=chrome_options)
        browser.set_window_size(window_width, window_height)
        return browser
    
    def _get_web_element_text(self) -> Tuple[List[Any], str]:
        """获取页面上所有可交互元素的文本"""
        web_elements = []
        elements_text = []
        
        # 获取所有可交互元素
        for tag in ['button', 'a', 'input', 'textarea', 'select']:
            elements = self.browser.find_elements(By.TAG_NAME, tag)
            for element in elements:
                try:
                    if element.is_displayed():
                        web_elements.append(element)
                        
                        # 获取元素文本
                        element_text = element.text.strip()
                        if not element_text:
                            element_text = element.get_attribute('value') or \
                                         element.get_attribute('placeholder') or \
                                         element.get_attribute('aria-label') or \
                                         element.get_attribute('title') or \
                                         f"<{tag}>"
                        
                        elements_text.append(f"[{len(web_elements)-1}] {element_text}")
                except:
                    continue
        
        return web_elements, "\n".join(elements_text)
    
    def _get_page_info(self, url: str, iteration: int) -> dict:
        """获取页面信息"""
        self.browser.get(url)
        time.sleep(5)  # 等待页面加载
        
        # 获取accessibility tree
        browser_info = fetch_browser_info(self.browser)
        accessibility_tree = fetch_page_accessibility_tree(
            browser_info,
            self.browser,
            current_viewport_only=True
        )
        tree_str, nodes_info = parse_accessibility_tree(accessibility_tree)
        clean_tree = clean_accesibility_tree(tree_str)
        
        # 获取页面元素文本
        web_elements, elements_text = self._get_web_element_text()
        
        # 保存截图
        screenshot_path = os.path.join(self.output_dir, f'screenshot_{iteration}.png')
        self.browser.save_screenshot(screenshot_path)
        
        return {
            'accessibility_tree': clean_tree,
            'screenshot_path': screenshot_path,
            'nodes_info': nodes_info,
            'web_elements': web_elements,
            'elements_text': elements_text
        }
    
    async def _analyze_page_content(self, node: PageNode, query: str) -> float:
        """使用LLM分析页面内容和相关性"""
        # 构造系统提示
        system_prompt = """Analyze the webpage content and determine its relevance to the user's query.
        Rate the relevance on a scale of 0.0 to 1.0, where 1.0 indicates highly relevant content.
        Consider both the accessibility tree structure and visual elements in the screenshot."""
        
        # 构造用户提示
        user_prompt = f"""Query: {query}
        
        Accessibility Tree:
        {node.accessibility_tree}
        
        Interactive Elements:
        {node.web_elements_text}
        
        Please analyze the content and provide:
        1. A relevance score (0.0-1.0)
        2. A brief explanation of why this page is or isn't relevant
        
        Format: <score>|<explanation>"""
        
        # 如果有截图，添加到消息中
        messages = [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_prompt}
        ]
        
        if node.screenshot_path:
            b64_img = encode_image(node.screenshot_path)
            messages[1]['content'] = [
                {'type': 'text', 'text': user_prompt},
                {'type': 'image_url', 'image_url': {'url': f"data:image/jpeg;base64,{b64_img}"}}
            ]
        
        # 调用LLM API
        response = await self.llm_client.chat.completions.create(
            model="gpt-4-vision-preview",
            messages=messages,
            max_tokens=1000,
            temperature=0.7
        )
        
        # 解析响应
        result = response.choices[0].message.content
        score_str, explanation = result.split('|', 1)
        relevance_score = float(score_str.strip())
        
        logging.info(f"Page relevance score: {relevance_score}")
        logging.info(f"Explanation: {explanation.strip()}")
        
        return relevance_score
    
    def _extract_links(self, nodes_info: dict) -> List[str]:
        """从accessibility tree中提取链接"""
        links = []
        for node in nodes_info:
            if node.get('tag_name') == 'a' and node.get('href'):
                href = node['href']
                if href.startswith('http'):
                    links.append(href)
        return list(set(links))
    
    async def explore(self, start_url: str, query: str):
        """开始网页探索"""
        self.start_time = time.time()
        self.search_tree = SearchTree(start_url, self.max_depth)
        iteration = 0
        
        while self._should_continue():
            iteration += 1
            current_node = self.search_tree.current_node
            
            if not current_node.is_visited:
                # 访问和分析页面
                page_info = self._get_page_info(current_node.url, iteration)
                current_node.accessibility_tree = page_info['accessibility_tree']
                current_node.screenshot_path = page_info['screenshot_path']
                current_node.web_elements_text = page_info['elements_text']
                
                # 使用LLM分析页面内容
                relevance_score = await self._analyze_page_content(current_node, query)
                current_node.relevance_score = relevance_score
                
                if relevance_score >= self.relevance_threshold:
                    current_node.has_useful_info = True
                    self.search_tree.useful_pages.append(current_node)
                    
                    # 提取和评估链接
                    links = self._extract_links(page_info['nodes_info'])
                    current_node.available_links = links
                    
                    # 将新链接添加到优先级队列
                    for link in links:
                        if link not in self.search_tree.visited_urls:
                            child_node = self.search_tree.add_child(current_node, link)
                            # 预估链接相关性
                            estimated_relevance = self._estimate_link_relevance(link, query)
                            self.search_tree.link_queue.put((-estimated_relevance, child_node))
                
                current_node.is_visited = True
                current_node.visit_time = datetime.now()
                self.search_tree.visited_urls.add(current_node.url)
            
            # 选择下一个要访问的节点
            if not self.search_tree.link_queue.empty():
                _, next_node = self.search_tree.link_queue.get()
                self.search_tree.current_node = next_node
            else:
                # 回溯
                next_node = self.search_tree.backtrack()
                if next_node is None:
                    break
    
    def _estimate_link_relevance(self, link: str, query: str) -> float:
        """估计链接的相关性（简单实现）"""
        # 将查询和链接转换为小写进行比较
        link_lower = link.lower()
        query_terms = query.lower().split()
        
        # 计算查询词在链接中出现的比例
        matching_terms = sum(1 for term in query_terms if term in link_lower)
        relevance = matching_terms / len(query_terms)
        
        return relevance
    
    async def generate_summary(self, query: str) -> str:
        """生成搜索结果摘要"""
        if not self.search_tree.useful_pages:
            return "No relevant information found."
        
        # 构建摘要提示
        summary_prompt = f"""Query: {query}
        
        Found {len(self.search_tree.useful_pages)} relevant pages. Please provide a comprehensive summary of the findings.
        
        Pages (sorted by relevance):
        """
        
        # 添加每个相关页面的信息
        sorted_pages = sorted(self.search_tree.useful_pages, 
                            key=lambda x: x.relevance_score, 
                            reverse=True)
        
        for i, page in enumerate(sorted_pages[:5], 1):  # 只使用前5个最相关的页面
            summary_prompt += f"\nPage {i} (Score: {page.relevance_score}):\n"
            summary_prompt += f"URL: {page.url}\n"
            summary_prompt += f"Content:\n{page.accessibility_tree[:1000]}...\n"  # 限制长度
        
        messages = [
            {'role': 'system', 'content': "Generate a concise but comprehensive summary of the search results."},
            {'role': 'user', 'content': summary_prompt}
        ]
        
        response = await self.llm_client.chat.completions.create(
            model="gpt-4",
            messages=messages,
            max_tokens=1000,
            temperature=0.7
        )
        
        return response.choices[0].message.content

async def main():
    # OpenAI client
    client = OpenAI(api_key='api_key', base_url='https://api.pumpkinaigc.online/v1')
    
    agent = WebAgent(
        llm_client=client,
        output_dir='search_results',
        relevance_threshold=0.7,
        max_depth=3,
        max_pages=50,
        time_limit=3600
    )
    
    try:
        query = "?"
        await agent.explore("https://www.example.com", query)
        summary = await agent.generate_summary(query)
        print(summary)
    finally:
        agent.cleanup()

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())