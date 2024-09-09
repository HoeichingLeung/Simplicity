from Simplicity.src.main import *
from Simplicity.utils.gpt_api import GPTclient
from openai import OpenAI
import re
import pandas as pd
import cProfile  
import pstats  
import io  
import ast

class AgentAPI(GPTclient):

    def __init__(self, api_key: str, base_url: str, csv_file_path: str):  
        """  
        :param api_key: API密钥  
        :param base_url: 中转url  
        :param csv_file_path: CSV文件路径  
        """  
        super().__init__(api_key, base_url)  
        self.csv_file_path = csv_file_path  # Save path as instance variable 
        self.university_data = pd.read_csv(self.csv_file_path)  
        self.has_greeted = False
        
    def query_professors_details(self, professor_name: str) -> str:  
        # 查找匹配的教授信息  
        matches = self.university_data[  
            self.university_data['Faculty'].str.strip().str.contains(professor_name, case=False, na=False)  
        ]  

        if not matches.empty:  
            # 提取相关列  
            selected_columns = matches[['Website', 'Email', 'full_research']]
            # selected_columns = matches[['Website', 'Email', 'full_research', 'publications']]  

            # 将数据转换为无序列表格式  
            response_content = selected_columns.apply(  
                lambda row: (  
                    f"- Website: {row['Website']}\n"  
                    f"- Email: {row['Email']}\n"  
                    f"- Full Research: {row['full_research']}\n"  
                    #f"- Publications: {row['publications']}\n"  
                ),  
                axis=1  
            ).to_list()  

            # 将列表转换为字符串，每个条目换行  
            response_content = "\n".join(response_content)  

            # 使用 self.get_response 生成回答  
            response = self.get_response(response_content)  
            print(response)  
            #return response  
        else:  
            print("No detailed information about this professor.")  
            #return "No data"

    def query_university_info(self, criteria: str) -> list:  
        #self.university_data = pd.read_csv(self.csv_file_path)
        # 提取排名区间  
        try:  
            min_rank, max_rank = map(int, criteria.split('-'))  
        except ValueError:  
            print("Please use 'min-max' format.")  
            return []  

        # 根据排名区间过滤数据  
        matches = self.university_data[  
            (self.university_data['Rank'].astype(int) >= min_rank) &  
            (self.university_data['Rank'].astype(int) <= max_rank)  
        ]  
        matches = matches.drop_duplicates(subset='University')  
        # 只选择 University 和 Rank 列  
        if not matches.empty:  
            # 将匹配的数据转换为字符串  
            content = matches.apply(  
                lambda row: (  
                    f"Ranked {row['Rank']}, {row['University']}"  
                ),  
                axis=1  
            ).to_list()  

            # 将列表转换为字符串，每个条目换行  
            content_str = "\n".join(content)  
            # 增加提示信息  
            prompt = "Rewrite the information.\n"  
            # 将提示与内容结合  
            content_str = prompt + content_str  
        else:  
            content_str = "No data"  

        # 调用self.get_response并传入content_str  
        response = self.get_response(content_str)  
        print(response)  

        # 返回大学名称列表  
        university_list = matches['University'].to_list() if not matches.empty else []  
        return university_list

    
    def query_professors(self, university=None, research_area=None) -> str:  
        # 查找匹配的教授信息 
        matches = self.university_data[  
            (self.university_data['University'].str.strip().str.contains(university, case=False, na=False) if university else True) &  
            (self.university_data['Research'].str.strip().str.contains(research_area, case=False, na=False) if research_area else True)  
        ]   # 大小写不敏感
        
        if not matches.empty:  
            # 将匹配的数据转换为字符串  
            content = matches.apply(  
                lambda row: (  
                    f"Ranked {row['Rank']}, {row['University']} has a faculty member named {row['Faculty']}. "  
                    f"They are involved in {row['Research']}. More information can be found on their website: {row['Website']}. "  
                    f"Contact them via email: {row['Email']}."  
                    f"Full Research: {row['full_research']}."  
                ),  
                axis=1  
            ).to_list()  

            # 将列表转换为字符串，每个条目换行  
            content = "\n".join(content)
        else:
            content = "No professors in this case."
        
        # 返回prof列表  
        professor_list = matches['Faculty'].to_list() if not matches.empty else []  
        prompt = "Here are the professors"
        content = prompt + content
        response = self.get_response(content)
        print(response)
        # 刷新数据库
        # self.university_data = matches
        return professor_list
            

    def query_api(self, query):  
        # Get the response from the API  
        response = self.get_response(content=query)  
        #print(response)
        # Check if the response is empty  
        if not response:  
            # Return a default message if the response is empty  
            print("Please try again later or check your query.")  

        # Return the response if it's not empty  
        print(response) 
        
    def personalized_recommend(self, inputs):
        pass
    
    def generate_code_for_query(self, user_query, history):  

        Prompt = f"""  

            You are a helpful assistant. Based on the user's current input, generate Python code that calls the functions to fetch university or professor information. The user has the history query, you need to consider the continuity of the question. Remember: try to generate Python each time.

            User current input: '{user_query}'  
            User input history: '{history}'  

            Existing functions:  
            1. query_university_info(criteria): criteria is the string parameter, the format is "1-10" and do not use any other format. Return a university list based on criteria.  
            2. query_professors(university=None, research_area=None): Return a professors list at a specific university or within a specific research area. "university" is the string parameter for one specific university, the argument "research_area" is one of ['Quantum materials', 'quantum nanomaterials','Quantum Optics']  
            3. query_professors_details(professor_name): Return the detail information about the professors like their publications and more research information. This function is called when users want more information.
            4. query_api(query): Return information with chatgpt for query, the query is asked by the user. That means the query may be not relvent with functions above.
            5. personalized_recommend(inputs): Take the whole user current input as the 'inputs' parameter. Return the personalized recommendation according to user's own information.

        """  

        # 调用OpenAI的GPT API生成代码  
        response = self.client.chat.completions.create(  
            messages=[{"role": "system", "content": Prompt}],  
            model="gpt-3.5-turbo",  
            max_tokens=512,  
            temperature=0.1
        )  
        print(response)  
        # 提取生成的代码  
        full_content = response.choices[0].message.content.strip()  
        # 使用正则表达式找到三重反引号之间的文本  
        code_match = re.search(r"```python\n(.*?)\n```", full_content, re.DOTALL)  

        if code_match:  
            generated_code = code_match.group(1)  # 提取代码部分  
            #print(generated_code)  
        else:  
            generated_code = full_content  # 将响应的完整文本作为生成代码
            #print("No Python code block found in the response.")  
        return generated_code  

    def exec_code(self, code_from_gpt):  
        # 提供安全执行环境  
        safe_globals = {  
            "query_api": self.query_api,  # 确保加入query_api  
            "query_professors": self.query_professors,  # 加入query_professor_info  
            "query_university_info": self.query_university_info,  # 加入query_university_info  
            "query_professors_details": self.query_professors_details,  
            "personalized_recommend": self.personalized_recommend,
            "print": print,  
            "__builtins__": {}  # 可选：限制内建函数以增强安全性  
        }  

        try:  
            # 执行生成的代码  
            exec(code_from_gpt, safe_globals)  
        except Exception as e:  
            print(f"执行代码时发生错误: {e}")  
    def greet_user(self):  
        if not self.has_greeted:  
            print("Welcome to the PhD Assistant Chatbot!\nType 'exit' to quit.")  
            self.has_greeted = True

def profile_code(code_to_run):  
    # 创建一个cProfile分析器  
    pr = cProfile.Profile()  
    pr.enable()  # 启用分析器  
    
    # 运行要分析的代码  
    code_to_run()  
    
    pr.disable()  # 禁用分析器  
    
    # 使用pstats处理输出并按照累计时间进行排序  
    s = io.StringIO()  
    sortby = 'cumulative'  # 可以使用 'tottime' 来按每个函数的执行时间排序  
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)  
    ps.print_stats(10)  # 可以调整数字以显示不同数量的前n个函数  

    # 打印分析结果  
    print(s.getvalue())  

def is_python_code(code):  
    try:  
        # 尝试解析代码字符串为 Python 语法树  
        ast.parse(code)  
        return True  
    except SyntaxError:  
        return False    

def run_agent_api():          
    # 设置 API 密钥和基础 URL  
    api_key = "sk-Erm52wwWJba3F2iz620d47D7F40e4fDcB2D36e9cC22bDe09"  
    base_url = "https://api.pumpkinaigc.online/v1"  

    # 创建 AgentAPI 实例  
    agent_api = AgentAPI(api_key, base_url, "output_with_codes.csv")  
    agent_api.greet_user() 

    # 初始化列表以存储用户查询  
    query_history = []  

    while True:  
        # 获取用户输入  
        user_query = input("Please enter your query: ")  

        # 如果用户输入 'exit'，则退出循环  
        if user_query.lower() == "exit":  
            print("Goodbye!")  
            break  

        # 确保历史记录最多存储10条记录  
        if len(query_history) >= 10:  
            query_history.pop(0)  # 移除最老的记录  

        # 添加当前查询到历史记录  
        query_history.append(user_query)  

        # 根据用户查询和查询历史生成代码  
        generated_code = agent_api.generate_code_for_query(user_query, query_history)  

        if generated_code:  
            if is_python_code(generated_code):  
                # 如果是 Python 代码，则执行  
                agent_api.exec_code(generated_code)  
            else:  
                # 否则，输出文本内容  
                print(generated_code)  
        else:  
            print("Sorry, I couldn't generate a response for that query. Please try again.") 

run_agent_api()       
