from Simplicity.src.main import *
from Simplicity.utils.gpt_api import GPTclient
class AgentAPI(GPTclient):

    def __init__(self, api_key: str, base_url: str):
        """
        :param api_key: API密钥
        :param base_url: 中转url
        """

        super().__init__(api_key, base_url)

    def query_university_info(self, criteria:str) -> str:
        pass

    def query_professor_info(self, university=None, research_area=None) -> str:
        pass

    def query_api(self, query):
        response = self.get_response(content=query)
        return response
    def generate_code_for_query(user_query):

        Prompt = f"""
            
            You are a helpful assistant. Based on the user's input, generate Python code that calls one of the functions to fetch university or professor information.
    
            User input: " {user_query}"
    
            Existing functions:
            1. query_university_info(criteria): Returns information about universities based on criteria like "top 10 in CS".
            2. query_professor_info(university=None, research_area=None): Returns information about professors at a specific university or within a specific research area. The argument"research_area" is one of ['chemistry', 'physics','computer science']
            3. query_api(query): Returns information with chatgpt for query, the query is asked by the user.
            """

        # 调用OpenAI的GPT API生成代码
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=Prompt,
            max_tokens=150,
            temperature=0
        )

        # 提取生成的代码
        generated_code = response.choices[0].text.strip()
        return generated_code


    def exec_code(code_from_gpt,query):
        user_query = query
        generated_code = generate_code_for_query(user_query)
        print("Generated Code:\n", generated_code)

        # 执行生成的代码，这里可能要验证消毒
        exec(generated_code)