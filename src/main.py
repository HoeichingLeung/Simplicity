from agent_api import AgentAPI 
import sys
sys.path.append("./utils") # 加入路径以便于直接运行
from gpt_api import GPTclient 
import numpy as np  
import streamlit as st  
from typing import List  
import pandas as pd 
import ast

def is_python_code(code):  
    try:  
        # 尝试解析代码字符串为 Python 语法树  
        ast.parse(code)  
        return True  
    except SyntaxError:  
        return False  

def run_agent_api_streamlit():  
    # Set API key and base URL  
    api_key = "sk-Erm52wwWJba3F2iz620d47D7F40e4fDcB2D36e9cC22bDe09"  
    base_url = "https://api.pumpkinaigc.online/v1"  

    # Map departments to their corresponding CSV files  
    department_to_csv = {  
        "Physics": "./data/physics_full.csv",   
    }  

    st.title("PhD Application Assistant")  

    # Initialize session state for messages  
    if "messages" not in st.session_state:  
        st.session_state.messages = []  
     

    # Sidebar for department selection  
    with st.sidebar:  
        st.header("Please choose your major:")  
        research_area = st.selectbox(  
            "Major:",  
            list(department_to_csv.keys())  
        ) 

    st.markdown("""  
    Welcome to your PhD Application Assistant! 🎓  

    **How can I assist you today?**  

    - **Select Your Major:** Use the dropdown menu on the left to choose your field of study. 
    - **Ask Questions:** Feel free to inquire about specific schools or professors to get detailed insights.  
    - **Personalized Recommendations:** Enter your personal information, and for the best results, include the phrase "personalized recommendations" in your query to receive customized advice and suggestions, powered by advanced language models.

    Let's embark on your academic journey together!  
    """)  

    # 显示历史消息  
    for msg in st.session_state.messages:  
        st.chat_message(msg["role"]).write(msg["content"])

    # Select the CSV file based on the selected department  
    csv_file_path = department_to_csv[research_area]  

    # Create AgentAPI instance  
    agent_api = AgentAPI(api_key, base_url, csv_file_path)  


    # Handle user input  
    if user_query := st.chat_input("Please input your question:"):  
        st.session_state.messages.append({"role": "user", "content": user_query})  
        st.chat_message("user").write(user_query)  

        # Generate response based on the query and CSV context  
        generated_code = agent_api.generate_code_for_query(user_query, st.session_state.messages)  

        if generated_code:  
            print(generated_code) # 在terminal，方便debug
            if is_python_code(generated_code):  
                # Execute Python code  
                response = agent_api.exec_code(generated_code)  
                #print(response)  
            else:  
                # Print text output  
                response = generated_code  
                st.session_state.messages.append({"role": "assistant", "content": response})  
                st.chat_message("assistant").write(response)
        else:  
            response = "Sorry, I couldn't generate a response for that query. Please try again."  
            st.session_state.messages.append({"role": "assistant", "content": response})  
            st.chat_message("assistant").write(response)


# Run the app  
if __name__ == "__main__":  
    run_agent_api_streamlit()