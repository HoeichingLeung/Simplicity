# 实现运行后对一个问题的online search
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, List, Any
from selenium import webdriver
from prompts import SYSTEM_PROMPT
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium import webdriver  
import time
import argparse
import logging
from openai import OpenAI
import os
from PageInteractionUtils import PageInteractionUtils
from accessibility_tree import fetch_browser_info, fetch_page_accessibility_tree,\
                    parse_accessibility_tree, clean_accesibility_tree
from webvoyager_utils import get_web_element_rect, encode_image, extract_information, print_message,\
    get_webarena_accessibility_tree, get_pdf_retrieval_ans_from_assistant, clip_message_and_obs, clip_message_and_obs_text_only

class PageNode:  
    """  
    表示搜索树中的单个网页节点  
    
    Attributes:  
        url (str): 页面URL  
        page_text (str): 从截图中提取的文本  
        accessibility_tree (str): 页面的accessibility tree内容  
        child_links (list): 待访问的子链接列表  
        is_visited (bool): 是否已访问过  
        is_useful (bool): 是否包含有用信息  
        parent (PageNode): 父节点  
        children (list): 子节点列表  
        depth (int): 节点在树中的深度  
        screenshot_b64 (str): 页面截图的base64编码  
    """  
    
    def __init__(self, url, parent=None):  
        # 基本属性  
        self.url = url  
        self.page_text = None  
        self.accessibility_tree = None  
        self.screenshot_path = None  
        
        # 节点关系  
        self.parent = parent  
        self.children = []  
        self.child_links = []  
        self.depth = 0 if parent is None else parent.depth + 1  
        
        # 状态标记  
        self.is_visited = False  
        self.is_useful = False  

    def add_child(self, child_node):  
        """  
        添加子节点  
        
        Args:  
            child_node (PageNode): 要添加的子节点  
        """  
        self.children.append(child_node)  
        child_node.parent = self  

    def update_content(self, page_text, accessibility_tree, screenshot_b64):  
        """  
        更新节点的页面内容  
        
        Args:  
            page_text (str): 页面文本内容  
            accessibility_tree (str): 可访问性树内容  
            screenshot_b64 (str): 截图的base64编码  
        """  
        self.page_text = page_text  
        self.accessibility_tree = accessibility_tree  
        self.screenshot_b64 = screenshot_b64  
        self.is_visited = True  

    def __str__(self):  
        """  
        节点的字符串表示  
        """  
        return f"PageNode(url={self.url}, depth={self.depth}, visited={self.is_visited}, useful={self.is_useful})"

class SearchTree:  
    """  
    管理网页搜索树的结构和搜索过程  
    
    Attributes:  
        root (PageNode): 根节点  
        max_depth (int): 最大搜索深度  
        visited_urls (set): 已访问URL集合  
        current_node (PageNode): 当前访问的节点  
    """  
    
    def __init__(self, initial_url, max_depth=3):  
        """初始化搜索树"""  
        pass  

    def expand_node(self, node, driver, gpt_client, query):  
        """  
        展开节点，访问其子链接  
        
        Args:  
            node (PageNode): 要展开的节点  
            driver: Selenium WebDriver实例  
            gpt_client: GPT API客户端  
            query (str): 搜索查询  
            
        Returns:  
            bool: 展开是否成功  
        """  
        pass  

    def backtrack(self):  
        """  
        回溯到上一个节点  
        
        Returns:  
            PageNode: 回溯后的当前节点  
        """  
        pass  

    def collect_useful_information(self):  
        """  
        收集树中所有有用的信息  
        
        Returns:  
            list: 包含有用信息的节点列表  
        """  
        pass  
  

class WebAgent:  
    def __init__(self, driver, openai_client, args):  
        """  
        初始化WebAgent  
        
        Args:  
            driver: 配置好的WebDriver实例  
            openai_client: OpenAI API客户端  
            args: 可选的其他配置参数  
        """  
        self.driver = driver  
        self.openai_client = openai_client  
        self.args = args
        self.utils = PageInteractionUtils()    
    
    # 好像跟哪里重复了我干
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
    
    def get_page_link(self, nodes_info: dict) -> List[str]:
        """从accessibility tree中提取链接/从页面中找到可下一步访问的链接？"""
        links = []
        for node in nodes_info:
            if node.get('tag_name') == 'a' and node.get('href'):
                href = node['href']
                if href.startswith('http'):
                    links.append(href)
        return list(set(links))
    
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


# 辅助函数  
def setup_webdriver(args):  
    """  
    设置和配置WebDriver  
    
    Args:  
        args: 命令行参数对象，包含浏览器配置选项  
        
    Returns:  
        WebDriver: 配置好的WebDriver实例  
    """  
    options = webdriver.ChromeOptions()  
    
    # 如果需要保存可访问性树,强制设置设备缩放比例  
    if args.save_accessibility_tree:  
        args.force_device_scale = True  
    
    # 设置设备缩放比例  
    if args.force_device_scale:  
        options.add_argument("--force-device-scale-factor=1")  
        
    # 无头模式配置  
    if args.headless:  
        options.add_argument("--headless")  
        # 设置特定的User-Agent  
        options.add_argument(  
            "--user-agent=Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "  
            "(KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36"  
        )  
    
    # 设置下载选项  
    prefs = {  
        "download.default_directory": args.download_dir,  
        "plugins.always_open_pdf_externally": True  # PDF文件直接下载而不是在浏览器中打开  
    }  
    options.add_experimental_option("prefs", prefs)  
    
    # 创建并返回WebDriver实例  
    driver = webdriver.Chrome(options=options)  
    
    # 设置窗口大小  
    driver.set_window_size(args.window_width, args.window_height)  
    
    return driver    

def setup_gpt_client(args):  
    """  
    设置GPT API客户端  
    
    Args:  
        args: 命令行参数对象，包含API配置  
        
    Returns:  
        OpenAI: 配置好的GPT客户端  
    """  
    # 默认配置
    base_url = "https://api.pumpkinaigc.online/v1"  
    
    # 创建客户端实例  
    try:  
        client = OpenAI(  
            api_key=args.api_key,  
            base_url=base_url  
        )  
        
        # 测试连接  
        client.models.list()  
        return client  
    
    except Exception as e:  
        logging.error(f"Failed to setup GPT client: {str(e)}")  
        raise    

def main(query):
    # 基本配置
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_file', type=str, default='data/test.json')
    parser.add_argument('--max_iter', type=int, default=3) # 最大迭代次数，好像算深度 
    parser.add_argument("--api_key", default="key", type=str, help="YOUR_OPENAI_API_KEY")
    parser.add_argument("--api_model", default="gpt-4-vision-preview", type=str, help="api model name")
    parser.add_argument("--output_dir", type=str, default='results')
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max_attached_imgs", type=int, default=1)
    parser.add_argument("--relevance_threshold", type=float, default=0.7)
    
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--download_dir", type=str, default="downloads")
    parser.add_argument("--text_only", action='store_true')
    # for web browser (setup_webdriver)
    parser.add_argument("--headless", action='store_true', help='The window of selenium')
    parser.add_argument("--save_accessibility_tree", action='store_true')
    parser.add_argument("--force_device_scale", action='store_true')
    parser.add_argument("--window_width", type=int, default=1024)
    parser.add_argument("--window_height", type=int, default=768)  # for headless mode, there is no address bar

    args = parser.parse_args()

    try:  
        driver = setup_webdriver(args)  
        client = setup_gpt_client(args)
        # 初始化搜索树和根节点  
        search_tree = SearchTree()  
        root_node = PageNode(  
            url="https://www.google.com",  
            title="Google Search",  
            depth=0,  # 根节点深度为0  
            parent=None  
        )  
        search_tree.add_node(root_node)  
        agent = WebAgent(driver, client, args)    

        
        max_depth = 3  # 设置最大搜索深度
        # Save Result file
        current_time = time.strftime("%Y%m%d_%H_%M_%S", time.localtime())
        result_dir = os.path.join(args.output_dir, current_time)
        os.makedirs(result_dir, exist_ok=True)
 
        try:  
            # 执行初始Google搜索  
            driver.get("https://www.google.com")  
            search_box = driver.find_element(By.NAME, "q")  
            agent.utils.exec_action_search(query, search_box, driver)  
            
            # 更新根节点信息  
            root_node.update_content(  
                title=driver.title,  
                url=driver.current_url  
            )  
            
            # 初始化搜索栈  
            search_stack = [(root_node, [])]  
            visited_urls = set([root_node.url])  
            it = 0            
            accumulate_prompt_token = 0
            accumulate_completion_token = 0
            while search_stack and it < args.max_iter:  
                logging.info(f'Iter: {it}')  
                it += 1  
                
                current_node, remaining_links = search_stack[-1]  
                fail_obs = ""  
                pdf_obs = ""  
                warn_obs = ""  
                
                try:  
                    # 获取页面信息  
                    rects, web_eles, web_eles_text = get_web_element_rect(driver, fix_color=args.fix_box_color)  
                    
                    # 获取accessibility tree  
                    accessibility_tree_path = os.path.join(task_dir, f'accessibility_tree{it}')  
                    ac_tree, obs_info = get_webarena_accessibility_tree(driver, accessibility_tree_path)  
                    
                    # 保存截图  
                    task_dir = os.path.join(result_dir)  
                    os.makedirs(task_dir, exist_ok=True)  
                    img_path = os.path.join(task_dir, f'screenshot{it}.png')  
                    driver.save_screenshot(img_path)  
                    
                    # 更新当前节点信息  
                    current_node.accessibility_tree = ac_tree  
                    current_node.screenshot_path = img_path  
                    current_node.page_text = web_eles_text  
                    
                    # 分析页面内容相关性  
                    page_content = {  
                        'accessibility_tree': ac_tree,  
                        'web_elements_text': web_eles_text,  
                        'url': driver.current_url,  
                        'title': driver.title  
                    }  
                    
                    # 使用agent分析页面相关性  
                    relevance_score = agent._analyze_page_content(page_content, query)  
                    current_node.relevance_score = relevance_score  
                    
                    # 编码图片并格式化消息  
                    b64_img = encode_image(img_path)  
                    curr_msg = PageInteractionUtils.format_msg(it, query, pdf_obs, warn_obs, b64_img, web_eles_text, ac_tree)  
                    messages.append(curr_msg)  
                    
                    # 裁剪消息历史  
                    messages = clip_message_and_obs(messages, args.max_attached_imgs)  
                    
                    # 调用GPT-4V API  
                    prompt_tokens, completion_tokens, gpt_call_error, openai_response = PageInteractionUtils.call_gpt4v_api(args, client, messages)  
                    
                    if not gpt_call_error:  
                        accumulate_prompt_token += prompt_tokens  
                        accumulate_completion_token += completion_tokens  
                        gpt_4v_res = openai_response.choices[0].message.content  
                        messages.append({'role': 'assistant', 'content': gpt_4v_res})  
                        
                        # 解析GPT响应，获取下一步行动  
                        action_key, info = extract_information(gpt_4v_res)  
                        
                        # 执行动作并更新搜索树  
                        if action_key == 'click':  
                            if not remaining_links:  
                                search_stack.pop()  
                                continue  
                            
                            next_link = remaining_links.pop()  
                            if current_node.depth >= max_depth:  
                                continue  
                            
                            # 创建新节点  
                            new_node = PageNode(  
                                url=next_link['url'],  
                                title=next_link['title'],  
                                depth=current_node.depth + 1,  
                                parent=current_node  
                            )  
                            
                            # 只有当页面相关性达到阈值时才继续搜索  
                            if relevance_score >= args.relevance_threshold:  
                                if new_node.url not in visited_urls:  
                                    search_tree.add_node(new_node)  
                                    current_node.add_child(new_node)  
                                    visited_urls.add(new_node.url)  
                                    
                                    # 获取新页面的链接  
                                    new_links = agent.get_page_links(driver)  
                                    # 预估链接相关性  
                                    scored_links = []  
                                    for link in new_links:  
                                        if link['url'] not in visited_urls:  
                                            estimated_relevance = agent._estimate_link_relevance(link, query)  
                                            scored_links.append((estimated_relevance, link))  
                                    
                                    # 按相关性排序链接  
                                    scored_links.sort(reverse=True)  
                                    unvisited_links = [link for score, link in scored_links]  
                                    
                                    if unvisited_links:  
                                        search_stack.append((new_node, unvisited_links))  
                            else:  
                                # 如果页面相关性低，回溯到上一个节点  
                                search_stack.pop()  
                                
                        elif action_key == 'wait':  
                            time.sleep(5)  
                        elif action_key == 'scroll':  
                            PageInteractionUtils.exec_action_scroll(info, web_eles, driver, args, None)  
                        elif action_key == 'answer':  
                            logging.info(info['content'])  
                            logging.info('finish!!')  
                            break  
                        else:  
                            raise NotImplementedError  
                        fail_obs = ""  
                            
                except Exception as e:  
                    logging.error('driver error info:')  
                    logging.error(e)  
                    if 'element click intercepted' not in str(e):  
                        fail_obs = "The action you have chosen cannot be executed. Please double-check if you have selected the wrong Numerical Label or Action or Action format. Then provide the revised Thought and Action."  
                    else:  
                        fail_obs = ""  
                    time.sleep(2)  
                    
                time.sleep(1)   
            
            # 生成最终摘要  
            summary = agent.generate_summary(query, search_tree)  
            #search_tree.save_to_file(f"search_results_{query[:30]}.json")  
            #print_message(messages, task_dir)
            driver.quit()
        
            logging.info(f'Total cost: {accumulate_prompt_token / 1000 * 0.01 + accumulate_completion_token / 1000 * 0.03}')  
            
        except Exception as e:  
            logging.error(f"Search tree generation failed: {e}")  
            
    finally:  
        if 'driver' in locals():  
            driver.quit()  

if __name__ == "__main__":
    query = "What is ..." 
    main(query)