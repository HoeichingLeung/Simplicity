from Simplicity.src.main import *
from Simplicity.utils.gpt_api import GPTclient
import re
import pandas as pd

class AgentAPI(GPTclient):

    def __init__(self, api_key: str, base_url: str, csv_file_path: str):  
        """  
        :param api_key: API密钥  
        :param base_url: 中转url  
        :param csv_file_path: CSV文件路径  
        """  
        super().__init__(api_key, base_url)  
        self.university_data = pd.read_csv(csv_file_path)  

    def query_university_info(self, criteria: str) -> str:  
        matching_universities = self.university_data[  
            self.university_data.apply(lambda row: criteria.lower() in row.to_string().lower(), axis=1)  
        ]  
        return matching_universities.to_string() if not matching_universities.empty else "No match found." 

    def query_professor_info(self, university=None, research_area=None) -> str:  
        # 查找匹配的教授信息  
        matches = self.university_data[  
            (self.university_data['University'].str.contains(university, na=False) if university else True) &  
            (self.university_data['Research'].str.contains(research_area, na=False) if research_area else True)  
        ]  

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
            # 增加提示信息  
            prompt = "Rewrite the information.\n"  
            # 将提示与内容结合  
            content = prompt + content 
        else:
            content = "No data"
            # 调用self.get_response并传入content  
        response = self.get_response(content)  
        return response
            

    def query_api(self, query):
        response = self.get_response(content=query)
        return response
    def generate_code_for_query(self, user_query):

        Prompt = f"""
            
            You are a helpful assistant. Based on the user's input, generate Python code that calls one of the functions to fetch university or professor information and print out the answer.
    
            User input: " {user_query}"
    
            Existing functions:
            1. query_university_info(criteria): Returns information about universities based on criteria like "top 10 in CS".
            2. query_professor_info(university=None, research_area=None): Returns information about professors at a specific university or within a specific research area. The argument"research_area" is one of ['Quantum materials', 'quantum nanomaterials','Quantum Optics']
            3. query_api(query): Returns information with chatgpt for query, the query is asked by the user.
            """

        # 调用OpenAI的GPT API生成代码
        response = self.client.chat.completions.create(  
            messages=[{"role": "system", "content": Prompt}],  
            model="gpt-3.5-turbo",  
            max_tokens=150,  
            temperature=0  
        )  

        # 提取生成的代码
        full_content = response.choices[0].message.content.strip()
        # Use a regular expression to find text between triple backticks (`)  
        code_match = re.search(r"```python\n(.*?)\n```", full_content, re.DOTALL)  

        if code_match:  
            generated_code = code_match.group(1)  # Extract the code part  
            print(generated_code)  
        else:  
            print("No Python code block found in the response.")  
        return generated_code


    def exec_code(self, code_from_gpt):  
        # 提供安全执行环境  
        safe_globals = {  
            "query_api": self.query_api,  # 确保加入query_api  
            "query_professor_info": self.query_professor_info,  # 加入query_professor_info  
            "query_university_info": self.query_university_info,  # 加入query_university_info  
            "print": print,
            "__builtins__": {}  # 可选：限制内建函数以增强安全性  
        }  

        try:  
            # 执行生成的代码  
            exec(code_from_gpt, safe_globals)  
        except Exception as e:  
            print(f"执行代码时发生错误: {e}")
        
if __name__ == "__main__":  
    # 设置API密钥和中转URL  
    api_key = "sk-Erm52wwWJba3F2iz620d47D7F40e4fDcB2D36e9cC22bDe09"  
    base_url = "https://api.pumpkinaigc.online/v1"  

    # 创建AgentAPI实例  
    agent_api = AgentAPI(api_key, base_url, "output_with_codes.csv")  
    
    # 根据用户提问生成代码  
    user_query = 'l want to know the professor research in quantum optics in Cornell?'  
    generated_code = agent_api.generate_code_for_query(user_query)  
    
    if generated_code:  
        # 执行生成的代码  
        agent_api.exec_code(generated_code)  
