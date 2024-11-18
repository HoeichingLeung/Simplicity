from dataclasses import dataclass, field
from prompts import SYSTEM_PROMPT
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.common.action_chains import ActionChains  
from selenium.webdriver.common.keys import Keys  
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
from datetime import datetime
import logging
from openai import OpenAI
import numpy as np


class PageInteractionUtils:  
    """  
    页面交互工具类，封装所有底层的页面操作和GPT交互功能  
    """  
    @staticmethod  
    def get_enhanced_prompt(is_first_iteration=True):  
        """生成增强的提示词"""  
        if is_first_iteration:  
            return (  
                "Please analyze this webpage using both visual and structural information:\n"  
                "1. VISUAL (Screenshot): Look for visual elements, layout, and design\n"  
                "2. STRUCTURAL (Accessibility Tree): Examine the page's semantic structure\n"  
                "3. CORRELATION: Cross-reference visual and structural information\n\n"  
                "Focus on:\n"  
                "- Interactive elements (buttons, forms, links)\n"  
                "- Content hierarchy and organization\n"  
                "- Navigation elements and their accessibility\n"  
                "- Any discrepancies between visual and structural representation"  
            )  
        else:  
            return (  
                "Based on the previous interaction and current page state:\n"  
                "1. Review the observation\n"  
                "2. Analyze both visual and structural changes\n"  
                "3. Determine the next appropriate action"  
            )

  

    @staticmethod  
    def format_msg(it, init_msg, pdf_obs, warn_obs, web_img_b64, web_text, ac_tree):  
        base_prompt = PageInteractionUtils.get_enhanced_prompt(is_first_iteration=(it == 1))  
        
        if it == 1:  
            full_msg = (  
                f"{init_msg}\n\n{base_prompt}\n\n"  
                f"Web Elements Information:\n{web_text}\n\n"  
                f"Accessibility Tree:\n{ac_tree}"  
            )  
        else:  
            observation = pdf_obs if pdf_obs else warn_obs  
            full_msg = (  
                f"Observation: {observation}\n\n{base_prompt}\n\n"  
                f"Web Elements Information:\n{web_text}\n\n"  
                f"Accessibility Tree:\n{ac_tree}"  
            )  
        
        return {  
            'role': 'user',  
            'content': [  
                {'type': 'text', 'text': full_msg},  
                {'type': 'image_url', 'image_url': {"url": f"data:image/png;base64,{web_img_b64}"}}  
            ]  
        }

  

    @staticmethod  
    def call_gpt4v_api(args, openai_client, messages):  
        """  
        调用 GPT-4 或 GPT-4V API 的函数，包含错误处理和重试机制  
        
        Args:  
            args: 包含配置参数的对象  
            openai_client: OpenAI API 客户端  
            messages: 发送给 API 的消息列表  
        
        Returns:  
            tuple: (prompt_tokens, completion_tokens, gpt_call_error, openai_response)  
        """  
        retry_times = 0  # 重试计数器  
        
        while True:  
            try:  
                # 根据模式选择调用不同的 API  
                if not args.text_only:  # 使用 GPT-4V（带视觉功能）  
                    logging.info('Calling gpt4v API...')  
                    openai_response = openai_client.chat.completions.create(  
                        model=args.api_model,  
                        messages=messages,  
                        max_tokens=1000,  
                        seed=args.seed  
                    )  
                else:  # 使用普通 GPT-4  
                    logging.info('Calling gpt4 API...')  
                    openai_response = openai_client.chat.completions.create(  
                        model=args.api_model,  
                        messages=messages,  
                        max_tokens=1000,  
                        seed=args.seed,  
                        timeout=30  # 文本模式设置超时时间  
                    )  

                # 获取 token 使用情况  
                prompt_tokens = openai_response.usage.prompt_tokens  
                completion_tokens = openai_response.usage.completion_tokens  

                # 记录 token 使用情况  
                logging.info(f'Prompt Tokens: {prompt_tokens}; Completion Tokens: {completion_tokens}')  

                # 调用成功  
                gpt_call_error = False  
                return prompt_tokens, completion_tokens, gpt_call_error, openai_response  

            except Exception as e:  
                logging.info(f'Error occurred, retrying. Error type: {type(e).__name__}')  

                # 处理不同类型的错误  
                if type(e).__name__ == 'RateLimitError':  # 速率限制错误  
                    time.sleep(10)  # 等待10秒后重试  
                    
                elif type(e).__name__ == 'APIError':  # API 错误  
                    time.sleep(15)  # 等待15秒后重试  
                    
                elif type(e).__name__ == 'InvalidRequestError':  # 无效请求错误  
                    gpt_call_error = True  
                    return None, None, gpt_call_error, None  
                    
                else:  # 其他错误  
                    gpt_call_error = True  
                    return None, None, gpt_call_error, None  

            # 增加重试次数并检查是否达到上限  
            retry_times += 1  
            if retry_times == 10:  # 最多重试10次  
                logging.info('Retrying too many times')  
                return None, None, True, None
  

    @staticmethod  
    def exec_action_click(info, web_ele, driver_task):  
        """  
        执行点击操作   
        """  
        # 修改链接打开方式  
        driver_task.execute_script("arguments[0].setAttribute('target', '_self')", web_ele)  
        
        try:  
            # 点击元素  
            web_ele.click()  
            
            # 使用显式等待替代 time.sleep  
            wait = WebDriverWait(driver_task, 10)  
            # 等待页面加载完成  
            wait.until(  
                lambda driver: driver.execute_script('return document.readyState') == 'complete'  
            )  
        except Exception as e:  
            print(f"Click action failed: {str(e)}")  

    @staticmethod  
    def exec_action_search(query: str, web_ele, driver_task):  
        """  
        在Google搜索框中输入查询并执行搜索  
        
        Args:  
            query: 搜索查询内容  
            web_ele: Google搜索框元素  
            driver_task: WebDriver实例
        """  
        try:  
            # 确保是Google搜索框  
            if web_ele.get_attribute('name') != 'q':  # Google搜索框的name属性固定为'q'  
                raise ValueError("This is not a Google search input box")  
                
            # 清除搜索框内容  
            web_ele.clear()  
            
            # 输入搜索内容  
            web_ele.send_keys(query)  
            
            # 短暂等待以确保输入完成  
            time.sleep(0.5)  
            
            # 按回车执行搜索  
            web_ele.send_keys(Keys.RETURN)  
            
            # 等待搜索结果加载  
            WebDriverWait(driver_task, 10).until(  
                lambda driver: driver.execute_script('return document.readyState') == 'complete'  
            )  
            
        except Exception as e:  
            print(f"Google search failed: {str(e)}")  
            raise

   

    @staticmethod  
    def exec_action_scroll(info, web_eles, driver_task, args, obs_info):   
        """  
        执行滚动操作  
        """  
        scroll_ele_number = info['number']  
        scroll_content = info['content']  
        
        try:  
            # 窗口滚动  
            if scroll_ele_number == "WINDOW":  
                scroll_distance = args.window_height * 2 // 3  
                if scroll_content != 'down':  
                    scroll_distance = -scroll_distance  
                    
                # 使用平滑滚动  
                driver_task.execute_script(  
                    "window.scrollBy({behavior: 'smooth', top: arguments[0]});",  
                    scroll_distance  
                )  
                
            # 元素滚动  
            else:  
                # 获取目标元素  
                if not args.text_only:  
                    web_ele = web_eles[int(scroll_ele_number)]  
                else:  
                    element_box = obs_info[scroll_ele_number]['union_bound']  
                    center_x = element_box[0] + element_box[2] // 2  
                    center_y = element_box[1] + element_box[3] // 2  
                    web_ele = driver_task.execute_script(  
                        "return document.elementFromPoint(arguments[0], arguments[1]);",  
                        center_x, center_y  
                    )  
                
                # 确保元素可见和可交互  
                WebDriverWait(driver_task, 10).until(  
                    EC.element_to_be_clickable(web_ele)  
                )  
                
                # 执行滚动  
                actions = ActionChains(driver_task)  
                driver_task.execute_script("arguments[0].focus();", web_ele)  
                
                key = Keys.ARROW_DOWN if scroll_content == 'down' else Keys.ARROW_UP  
                actions.key_down(Keys.ALT).send_keys(key).key_up(Keys.ALT).perform()  
                
            # 等待滚动完成  
            WebDriverWait(driver_task, 10).until(  
                lambda driver: driver.execute_script('return document.readyState') == 'complete'  
            )  
                
        except Exception as e:  
            print(f"Scroll action failed: {str(e)}")  
            raise  
