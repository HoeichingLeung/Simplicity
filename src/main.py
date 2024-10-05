import sys
import pandas as pd
import streamlit as st
import ast
import os
import logging

sys.path.append("./utils")  # 加入路径以便于直接运行
from gpt_api import GPTclient
from agent_api import AgentAPI

# 配置日志记录
logging.basicConfig(level=logging.INFO)


def is_python_code(code):
    """检查代码是否为有效的Python代码段"""
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False


def load_csv_file(department, subfield=None):
    """根据部门名称和可能的子领域加载对应的CSV文件"""
    department_to_csv = {
        "Physics": "./data/major_data/physics_full.csv",
        "Mechanical Engineering": "./data/major_data/mechanical_engineering.csv",
        "Computer Science": {
            "AI": "./data/major_data/computer_science_ai.csv",
            "Systems": "./data/major_data/computer_science_systems.csv",
            "Theory": "./data/major_data/computer_science_theory.csv",
            "Interdisciplinary": "./data/major_data/computer_science_interdisciplinary.csv",
        },
    }

    if subfield is None:
        return department_to_csv.get(department)
    else:
        return department_to_csv.get(department, {}).get(subfield)


def run_agent_api_streamlit():
    """运行 Streamlit 应用"""

    # 设置 API 密钥和基础 URL
    api_key = "sk-Erm52wwWJba3F2iz620d47D7F40e4fDcB2D36e9cC22bDe09"
    base_url = "https://api.pumpkinaigc.online/v1"

    st.title("PhD Application Assistant")

    # 初始化会话状态的消息
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # 侧边栏选择部门
    with st.sidebar:
        st.header("Please choose your major:")
        department = st.selectbox(
            "Major:", ["Physics", "Mechanical Engineering", "Computer Science"]
        )

        subfield = None
        if department == "Computer Science":
            subfield = st.selectbox(
                "Specialization:", ["AI", "Systems", "Theory", "Interdisciplinary"]
            )

        csv_file_path = load_csv_file(department, subfield)

        if not csv_file_path:
            st.error("Information for the selected department is not available.")
            return

    st.markdown(
        """  
    Welcome to your PhD Application Assistant! 🎓  

    **How can I assist you today?**  

    - **Select Your Major:** Use the dropdown menu on the left to choose your field of study.  
    - **Ask Questions:** Feel free to inquire about specific schools or professors to get detailed insights.  
    - **Personalized Recommendations:** Enter your personal information to receive customized advice and suggestions powered by advanced language models.  

    Let's embark on your academic journey together!  
    """
    )

    # 显示历史消息
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    if csv_file_path is None:
        st.error("Information for the selected department is not available.")
        return

    agent_api = AgentAPI(api_key, base_url, csv_file_path)

    if user_query := st.chat_input("Please input your question:"):
        # 检查消息列表的长度，如果超过15条，则移除最早的一条
        if len(st.session_state.messages) >= 15:
            st.session_state.messages.pop(0)

        # 添加用户的查询到消息列表
        st.session_state.messages.append({"role": "user", "content": user_query})
        st.chat_message("user").write(user_query)

        try:
            generated_code = agent_api.generate_code_for_query(
                user_query, st.session_state.messages
            )
            if generated_code:
                logging.info(f"Generated code: {generated_code}")
                if is_python_code(generated_code):
                    response = agent_api.exec_code(generated_code)
                else:
                    response = generated_code
                    st.chat_message("assistant").write(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
            else:
                response = "Sorry, I couldn't generate a response for that query. Please try again."
        except Exception as e:
            logging.error(f"Error processing query: {e}")
            response = "An error occurred while processing your request."

        if len(st.session_state.messages) >= 15:
            st.session_state.messages.pop(0)

if __name__ == "__main__":
    run_agent_api_streamlit()
