import os
import sys
import logging
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
import ast
import re
import json
import torch

# 添加必要的路径
sys.path.append("./utils")

# Import custom modules
from gpt_api import GPTclient
from compute_embedding import EmbeddingModel

# 配置日志系统
logging.basicConfig(level=logging.INFO)


def is_python_code(code):
    """Check if the provided code is a valid Python snippet."""
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False


class AgentAPI(GPTclient):
    def __init__(self, api_key: str, base_url: str, csv_file_path: str):
        """
        Initialize the AgentAPI class.

        :param api_key: API Key for authentication
        :param base_url: Base URL for API requests
        :param csv_file_path: Path to the CSV file containing data
        """
        super().__init__(api_key, base_url)
        self.csv_file_path = csv_file_path
        try:
            self.university_data = pd.read_csv(self.csv_file_path)
        except FileNotFoundError as e:
            logging.error(f"CSV file not found: {e}")
            self.university_data = pd.DataFrame()

    def query_professors_details(self, professor_name: str, user_query: str) -> None:
        user_query += " Use the provided information to generate a clear and accurate response to the query."
        try:
            matches = self.university_data[
                self.university_data["Faculty"]
                .str.strip()
                .str.contains(professor_name, case=False, na=False)
            ]
            if not matches.empty:
                # Extract relevant columns
                response_content = (
                    matches[["Website", "Email", "full_research", "publications"]]
                    .apply(
                        lambda row: (
                            f"- Website: {row['Website']}\n"
                            f"- Email: {row['Email']}\n"
                            f"- Full Research: {row['full_research']}\n"
                            f"- Publications: {row['publications']}\n"
                        ),
                        axis=1,
                    )
                    .to_list()
                )

                content = "\n".join(response_content) + user_query
                response = self.get_response_prof_details(content)
            else:
                response = "No detailed information about this professor."

            st.chat_message("assistant").write(response)
            st.session_state.messages.append({"role": "assistant", "content": response})

        except KeyError as e:
            logging.error(f"Dataframe key error: {e}")

    def query_university_rank(self, criteria: str, user_query: str) -> list:
        user_query += " Use the provided information to generate a clear and accurate response to the query."
        try:
            min_rank, max_rank = map(int, criteria.split("-"))
            matches = self.university_data[
                (self.university_data["Rank"].astype(int) >= min_rank)
                & (self.university_data["Rank"].astype(int) <= max_rank)
            ].drop_duplicates(subset="University")

            if not matches.empty:
                content = matches.apply(
                    lambda row: f"Ranked {row['Rank']}, {row['University']}", axis=1
                ).to_list()
                content_str = "\n".join(content)
            else:
                content_str = "No data"

            content_str = user_query + content_str
            response = self.get_response(content_str)
            st.chat_message("assistant").write(response)
            st.session_state.messages.append({"role": "assistant", "content": response})

            return matches["University"].to_list()

        except ValueError:
            response = "Please use 'min-max' format."
            st.chat_message("assistant").write(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
            return []

    def query_professors(
        self, university_list=None, research_area=None, user_query=None
    ) -> list:
        user_query += " Use the provided information to generate a clear and accurate response to the query."
        professor_list = []
        content = ""
        all_matches = pd.DataFrame()

        for university in university_list or []:
            # 查找匹配的教授信息
            matches = self.university_data[
                (
                    self.university_data["University"].str.contains(
                        university, case=False, na=False
                    )
                    if university
                    else True
                )
                & (
                    self.university_data["Research"].str.contains(
                        research_area, case=False, na=False
                    )
                    if research_area
                    else True
                )
            ]

            if not matches.empty:
                university_content = matches.apply(
                    lambda row: (
                        f"Ranked {row['Rank']}, {row['University']} has a faculty member named {row['Faculty']}. "
                        f"They are involved in {row['Research']}. More information can be found on their website: {row['Website']}. "
                        f"Contact them via email: {row['Email']}. "
                        f"Full Research: {row['full_research']}."
                    ),
                    axis=1,
                ).to_list()

                professor_list.extend(matches["Faculty"].tolist())
                content += "\n".join(university_content) + "\n"
                all_matches = pd.concat([all_matches, matches], ignore_index=True)

        if all_matches.empty:
            content = "No professors."

        full_content = user_query + content
        response = self.get_response(full_content)

        st.chat_message("assistant").write(response)
        st.session_state.messages.append({"role": "assistant", "content": response})

        if not all_matches.empty:
            self.university_data = all_matches

        return professor_list

    def query_api(self, user_query):
        response = self.get_response_query(content=user_query)

        if not response:
            st.write("Please try again later or check your query.")
        else:
            st.chat_message("assistant").write(response)
            st.session_state.messages.append({"role": "assistant", "content": response})

    def personalized_recommendations(self, user_query: str):
        model_name = "BCEmbeddingmodel"  # 使用相对路径
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dilimitor = "###RAG###"

        model = EmbeddingModel(
            model_name=model_name,
            device=device,
            api_key=self.api_key,
            base_url=self.base_url,
        )
        query_embedding = model.get_embeddings([user_query])

        # 计算每个列名的embedding与query_embedding的余弦相似度
        similarities = []
        col_embed_path = "./data/embeddings/column_embeddings.npy"
        column_name_embeddings = np.load(col_embed_path)
        # print(column_embeddings.shape)
        for i, embedding in enumerate(column_name_embeddings):
            # 计算余弦相似度
            similarity = cosine_similarity(
                query_embedding.reshape(1, -1), embedding.reshape(1, -1)
            )[0][0]
            similarities.append(similarity)
            # print(f"Similarity for column {i}: {similarity}")

        # 归一化相似度
        max_sim = max(similarities)
        min_sim = min(similarities)
        normalized_similarities = [
            (sim - min_sim) / (max_sim - min_sim) for sim in similarities
        ]

        json_file_path = "data/embeddings/embeddings_files.json"

        # 打开并读取 JSON 文件
        with open(json_file_path, "r") as f:
            embeddings_files = json.load(f)
        num_rows = None
        scores = None
        weights = np.array(normalized_similarities)
        df = pd.read_csv(self.csv_file_path)
        # original_indices = df.index.tolist()  # 保存原始索引

        for i, file_name in enumerate(embeddings_files):
            # 构建完整文件路径
            embeddings_dir = "./data/embeddings/physics"
            file_path = os.path.join(embeddings_dir, file_name)

            # 检查文件是否存在
            if not os.path.exists(file_path):
                print(f"Warning: {file_path} does not exist.")
                continue

            # 加载对应列的嵌入数据
            embeddings = np.load(file_path)

            # 初始化 scores 数组
            if scores is None:
                num_rows = embeddings.shape[0]
                scores = np.zeros(num_rows)

            # 计算这一列的相似度
            similarity_scores = cosine_similarity(embeddings, query_embedding).flatten()

            # 作用权重于相似度
            weighted_similarity_scores = similarity_scores * weights[i]

            # 将加权相似度加到总分数
            scores += weighted_similarity_scores

        if scores is not None:
            # 找到相似度最高的3行
            top_3_indices = np.argsort(scores)[-3:][::-1]
            # print("Top 3 most similar rows indices and their texts:")
            # 用于存储三个结果
            rag_results = []
            print(top_3_indices)
            for index in top_3_indices:
                rag_result = ", ".join(df.iloc[index].astype(str))
                rag_results.append(rag_result)
        else:
            print("Error: No embeddings were loaded.")

        print(rag_results)
        content = user_query + dilimitor + str(rag_results)  # + str(web_result)
        response = self.get_response_psnl(content)

        # Return the response
        st.chat_message("assistant").write(response)
        st.session_state.messages.append({"role": "assistant", "content": response})

    def generate_code_for_query(self, user_query, history):

        Prompt = f"""  
        You are a helpful assistant tasked with generating Python code to fetch information about universities or professors based on the user's input. Each response should maintain continuity with the user's query history and focus on generating Python code that calls functions at a time. Use `personalized_recommendations` for queries that relate to personal academic interests, career aspirations, specific research goals, or specific reigons such as US or Europe.  
  
        User input history: '{history}'  

        Available functions:  
        1. query_api(user_query): Use this function to directly access information via ChatGPT for any query where specific functions fail to address the user's needs adequately. Ideal for broad inquiries or when the user's request does not fit within the constraints or capabilities of the available functions. Leverages ChatGPT's broad knowledge base to provide comprehensive answers without relying on predefined function parameters.  
        2. query_professors(university_list, research_area, user_query): Retrieves a list of professors at a particular university or within a specific research area. Example: university_list=['Harvard University', 'EPFL']. "Use this when asking 'who works' or 'which professor'."   
        3. query_professors_details(professor_name, user_query): Provides detailed information about a professor's publications and research. Used for deeper insights.  
        4. query_university_rank(criteria, user_query): Returns a list of universities based on the ranking criteria specified in the format "1-3". To translate user input such as "top N universities", convert to the format "1-N".  
        5. personalized_recommendations(user_query): Use this function when the user's query indicates a need for recommendations based on their specific academic interests, career goals, or personal aspirations. Ideal for queries that mention personal interests in research topics, such as "first-principles exploration of novel quantum physics," and requests for advice on what might suit their unique profile and objectives.   

        Guidelines:  
        1. Remember to take the user current input as an input parameter for each function. 
        2. If the user's input includes a specific university, include only that university in university_list.  
        3. Importing functions is not required.  
        4. Generate code for more than one function only when it is possible.  
        5. If the user's input pertains to personalized academic interests or aspirations (e.g., research areas, publication history, specific personal goals), use `personalized_recommendations`.
        6. Use `query_api` for more general questions or inquiries unrelated to academic and research personalizations.  
        7. If generate more than one function, ensure that the final function call directly executes the operation (e.g., by printing the result) instead of storing it in a variable. 
        """

        # Modify the messages part to properly pass the prompt to the API
        response = self.client.chat.completions.create(
            messages=[
                {"role": "system", "content": Prompt},
                {"role": "user", "content": user_query},
            ],
            model="gpt-3.5-turbo",
            max_tokens=512,
            temperature=0.1,
        )

        full_content = response.choices[0].message.content.strip()
        code_match = re.search(r"```python\n(.*?)\n```", full_content, re.DOTALL)

        return code_match.group(1) if code_match else full_content

    def exec_code(self, code_from_gpt):
        safe_globals = {
            "query_api": self.query_api,
            "query_professors": self.query_professors,
            "query_university_rank": self.query_university_rank,
            "query_professors_details": self.query_professors_details,
            "personalized_recommendations": self.personalized_recommendations,
            "print": print,
            "__builtins__": {},
        }

        try:
            exec(code_from_gpt, safe_globals)
        except Exception as e:
            logging.error(f"Error executing code: {e}")


def run_agent_api():
    """Only a test for this part."""
    # 设置 API 密钥和基础 URL
    api_key = "sk-Erm52wwWJba3F2iz620d47D7F40e4fDcB2D36e9cC22bDe09"
    base_url = "https://api.pumpkinaigc.online/v1"

    # 创建 AgentAPI 实例
    agent_api = AgentAPI(api_key, base_url, "./data/major_data/physics_full.csv")

    # 初始化列表以存储用户查询
    query_history = []

    while True:
        user_query = input("Please enter your query: ")

        if user_query.lower() == "exit":
            logging.info("Exiting the application.")
            break

        if len(query_history) >= 10:
            query_history.pop(0)

        query_history.append(user_query)

        try:
            generated_code = agent_api.generate_code_for_query(
                user_query, query_history
            )
            if generated_code:
                if is_python_code(generated_code):
                    agent_api.exec_code(generated_code)
                else:
                    print(generated_code)
            else:
                print(
                    "Sorry, I couldn't generate a response for that query. Please try again."
                )
        except Exception as e:
            logging.error(f"An error occurred: {e}")


if __name__ == "__main__":
    run_agent_api()
