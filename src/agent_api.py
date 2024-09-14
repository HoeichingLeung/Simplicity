import sys
sys.path.append("./utils") # 加入路径以便于直接运行
sys.path.append("./scripts") # 加入路径以便于直接运行
from gpt_api import GPTclient  
from compute_embedding import EmbeddingModel, read_text_data, save_embeddings, save_mappings
from openai import OpenAI
import streamlit as st
import re
import pandas as pd 
import ast
import torch
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
        
    def query_professors_details(self, professor_name: str, user_query) -> str:  
        user_query = user_query + "Use the provided information to generate a clear and accurate response to the query."
        # 查找匹配的教授信息  
        matches = self.university_data[  
            self.university_data['Faculty'].str.strip().str.contains(professor_name, case=False, na=False)  
        ]  

        if not matches.empty:  
            # 提取相关列  
            selected_columns = matches[['Website', 'Email', 'full_research', 'publications']]  

            # 将数据转换为无序列表格式  
            response_content = selected_columns.apply(  
                lambda row: (  
                    f"- Website: {row['Website']}\n"  
                    f"- Email: {row['Email']}\n"  
                    f"- Full Research: {row['full_research']}\n"  
                    f"- Publications: {row['publications']}\n"  
                ),  
                axis=1  
            ).to_list()  

            # 将列表转换为字符串，每个条目换行  
            response_content = "\n".join(response_content)  
            content = response_content + user_query
            # 使用 self.get_response 生成回答  
            response = self.get_response(content)  
            
            st.chat_message("assistant").write(response) 
            st.session_state.messages.append({"role": "assistant", "content": response}) 
            #return response  
        else:  
            st.chat_message("assistant").write("No detailed information about this professor.") 
            st.session_state.messages.append({"role": "assistant", "content": response})  
            #return "No data"

    def query_university_rank(self, criteria: str, user_query) -> list:  
        user_query = user_query + "Use the provided information to generate a clear and accurate response to the query."
        #self.university_data = pd.read_csv(self.csv_file_path)
        # 提取排名区间  
        try:  
            min_rank, max_rank = map(int, criteria.split('-'))  
        except ValueError:  
            st.chat_message("assistant").write("Please use 'min-max' format.") 
            st.session_state.messages.append({"role": "assistant", "content": "Please use 'min-max' format."})
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
            prompt = user_query  
            # 将提示与内容结合  
            content_str = prompt + content_str  
        else:  
            content_str = "No data"  

        # 调用self.get_response并传入content_str  
        response = self.get_response(content_str)  
        #print(response)
        st.chat_message("assistant").write(response) 
        st.session_state.messages.append({"role": "assistant", "content": response}) 

        # 返回大学名称列表  
        university_list = matches['University'].to_list() if not matches.empty else []  
        return university_list

    
    def query_professors(self, university_list=None, research_area=None, user_query=None) -> list:  
        user_query = user_query + "Use the provided information to generate a clear and accurate response to the query."
        # Initialize an empty list to store professors  
        professor_list = []  
        # Initialize an empty string to accumulate content  
        content = ""  
        # Initialize an empty DataFrame to accumulate matches  
        all_matches = pd.DataFrame()  

        # Iterate over each university in the university_list  
        for university in university_list:  
            # Find matching professor information for each university  
            matches = self.university_data[  
                (self.university_data['University'].str.strip().str.contains(university, case=False, na=False) if university else True) &  
                (self.university_data['Research'].str.strip().str.contains(research_area, case=False, na=False) if research_area else True)  
            ]  # Case insensitive  

            if not matches.empty:  
                # Convert the matching data to a string  
                university_content = matches.apply(  
                    lambda row: (  
                        f"Ranked {row['Rank']}, {row['University']} has a faculty member named {row['Faculty']}. "  
                        f"They are involved in {row['Research']}. More information can be found on their website: {row['Website']}. "  
                        f"Contact them via email: {row['Email']}."  
                        f"Full Research: {row['full_research']}."  
                    ),  
                    axis=1  
                ).to_list()  

                # Add the professors to the list  
                professor_list.extend(matches['Faculty'].to_list())  

                # Append the university content to the overall content  
                content += "\n".join(university_content) + "\n"  

                # Accumulate matches  
                all_matches = pd.concat([all_matches, matches], ignore_index=True)  
            else:  
                matches = pd.DataFrame()  
        if all_matches.empty:
            content = 'No professors.'
        # Prepare the prompt and get the response  
        prompt = user_query 
        full_content = prompt + content  
        response = self.get_response(full_content)  

        st.chat_message("assistant").write(response) 
        st.session_state.messages.append({"role": "assistant", "content": response}) 
        if not all_matches.empty:
            # Refresh self.university_data with all matches  
            self.university_data = all_matches  

        # Return the list of professors  
        return professor_list
            

    def query_api(self, user_query):  
        # Get the response from the API  
        response = self.get_response(content=user_query)  
        #print(response)
        # Check if the response is empty  
        if not response:  
            # Return a default message if the response is empty  
            st.write("Please try again later or check your query.")  

        # Return the response if it's not empty  
        st.chat_message("assistant").write(response) 
        st.session_state.messages.append({"role": "assistant", "content": response}) 
        
    def personalized_recommendations(self, user_query=None):
        user_query = user_query + "Use the provided information to generate a clear and accurate response to the query."
        model_name = 'BCEmbeddingmodel'  # 使用相对路径
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        file_path_npy = './data/embeddings/physics_full_embedding.npy'
        model = EmbeddingModel(model_name=model_name, device=device)
        model.load_embed(file_path = file_path_npy)
        rag_result = model.query_for_sentences(k=3,question = user_query)
        #print(">rag:",rag_result)
        content = user_query + rag_result
        response = self.get_response(content)
        
        # Return the response 
        st.chat_message("assistant").write(response) 
        st.session_state.messages.append({"role": "assistant", "content": response}) 
    
    def generate_code_for_query(self, user_query, history):  

        Prompt = f"""  
            You are a helpful assistant tasked with generating Python code to fetch information about universities or professors based on the user's input. Each response should maintain continuity with the user's query history and focus on generating Python code that calls only one function at a time. If the user's input is unrelated to the functions query_university_rank, query_professors, query_professors_details, or personalized_recommendations, directly generate code using query_api(query). If user's input contains 'personalized', use function personalized_recommendations. 

            User current input: '{user_query}'  
            User input history: '{history}'  

            Available functions:  
            1. query_api(user_query): Use this function to directly access information via ChatGPT for any query where specific functions fail to address the user's needs adequately. Ideal for broad inquiries or when the user's request does not fit within the constraints or capabilities of the available functions. Leverages ChatGPT's broad knowledge base to provide comprehensive answers without relying on predefined function parameters.  
            2. query_professors(university_list, research_area, user_query): Retrieves a list of professors at a particular university or within a specific research area. Example: university_list=['Harvard University', 'EPFL']. "Use this when asking 'who works' or 'which professor'."   
            3. query_professors_details(professor_name, user_query): Provides detailed information about a professor's publications and research. Used for deeper insights.  
            4. query_university_rank(criteria, user_query): Returns a list of universities based on the ranking criteria specified in the format "1-3". To translate user input such as "top N universities", convert to the format "1-N".  
            5. personalized_recommendations(user_query): Use this function when the user's query indicates a need for recommendations based on their specific academic interests, career goals, or personal aspirations. Ideal for queries that mention personal interests in research topics, such as "first-principles exploration of novel quantum physics," and requests for advice on what might suit their unique profile and objectives. 

            Guidelines:  
            1. Remember to take the user current input 'user_query' as input parameter for each function.  
            2. If the user's input includes a specific university, include only that university in university_list.  
            3. Importing functions is not required.  
            4. Generate code for only one function at a time.  
            5. If a query requires multiple steps or cannot be satisfied by a single function call, clearly inform the user: Respond with, "I'm unable to process this request in one step. Please break it down into simpler, sequential questions." This response should guide the user to split their question into smaller, manageable parts that align with the available functions.  
            6. If user's input contains 'personalized', use function personalized_recommendations.
            7. Example for addressing complex queries:  
                User Query: "List top 10 universities and their professors in Quantum Optics."  
                Assistant Response: "I'm unable to process this request in one step. Please break it down into simpler, sequential questions."  
                Suggested Steps for the User:  
                    "List top 10 universities."  
                    "List professors in Quantum Optics at [specific university]."  
            """

        # 调用OpenAI的GPT API生成代码  
        response = self.client.chat.completions.create(  
            messages=[{"role": "system", "content": Prompt}],  
            model="gpt-3.5-turbo",  
            max_tokens=512,  
            temperature=0.1
        )  
        #print(response)  
        # 提取生成的代码  
        full_content = response.choices[0].message.content.strip()  
        # 使用正则表达式找到三重反引号之间的文本  
        code_match = re.search(r"```python\n(.*?)\n```", full_content, re.DOTALL)  

        if code_match:  
            generated_code = code_match.group(1)  # 提取代码部分  
            #print(generated_code)  
        else:  # 找不到代码
            generated_code = full_content  # 将响应的完整文本作为生成代码
            #print("No Python code block found in the response.")  
        return generated_code  

    def exec_code(self, code_from_gpt):  
        # 提供安全执行环境  
        safe_globals = {  
            "query_api": self.query_api,  # 确保加入query_api  
            "query_professors": self.query_professors,  # 加入query_professor_info  
            "query_university_rank": self.query_university_rank,  # 加入query_university_info  
            "query_professors_details": self.query_professors_details,  
            "personalized_recommendations": self.personalized_recommendations,
            "print": print,  
            "__builtins__": {}  # 可选：限制内建函数以增强安全性  
        }  

        try:  
            # 执行生成的代码  
            response = exec(code_from_gpt, safe_globals)  
            return response
        except Exception as e:  
            print(f"执行代码时发生错误: {e}")  


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
    agent_api = AgentAPI(api_key, base_url, "./data/physics_full.csv")  
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

if __name__ == '__main__':
    run_agent_api()       
