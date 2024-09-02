import torch  
import transformers  
import numpy as np  
import streamlit as st  
from typing import List  
import pandas as pd 
from io import StringIO
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM  
import tempfile  

# 加载配置文件  
with open('config.json', 'r') as config_file:  
    config = json.load(config_file)

# 文本切分类  
class TextSplitter:  
    def __init__(self, max_chunk_size=config['max_chunk_size'], overlap=config['overlap']):  
        self.max_chunk_size = max_chunk_size  
        self.overlap = overlap  

    def split_text(self, text: str):  
        words = text.split()  
        chunks = []  
        for i in range(0, len(words), self.max_chunk_size - self.overlap):  
            chunk = words[i:i + self.max_chunk_size]  
            chunks.append(' '.join(chunk))  
        return chunks   
    
class LLM:  
    def __init__(self, model_path: str = config['model_path']) -> None:  
        print("Creating tokenizer...")  
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)  

        print("Creating model...")  
        self.model = transformers.AutoModelForCausalLM.from_pretrained(  
            model_path,  
            torch_dtype=torch.bfloat16  
        ).cuda()  

        print(f'Loading Llama 3 model from {model_path}.')  

    def generate(self, question: str, context: list):  
        if context:  
            prompt = (  
                f'Background: {context}\n'  
                "I am currently applying for a Ph.D., "  
                "and the above background provides academic information about schools and professors. "  
                "Please answer my question: "  
                f"{question}\n<|start_of_answer|>"  
            )  
        else:  
            prompt = question + "\n<|start_of_answer|>"  

        inputs = self.tokenizer(prompt, return_tensors="pt")["input_ids"].cuda()  
        outputs = self.model.generate(  
            inputs,  
            do_sample=True,  
            max_new_tokens=256,  
            temperature=config['temperature'],  
            top_p=config['top_p']  
        )  
        # 解码模型的输出  
        output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)  

        # 分割输出以找到标记后的内容  
        result_after_marker = output.split("<|start_of_answer|>")[-1].strip()  

        # 移除可能存在的 <|end_of_answer|> 标记  
        if "<|end_of_answer|>" in result_after_marker:  
            result_after_marker = result_after_marker.split("<|end_of_answer|>")[0].strip()  

        return result_after_marker  

# 定义向量模型类  
class EmbeddingModel:  
    def __init__(self, model_name: str = config['embedding_model_path'], device: str = config['device']):  
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)  
        self.model = transformers.AutoModel.from_pretrained(model_name)  
        self.device = device  
        self.model.to(self.device)  

    def get_embeddings(self, sentences: List[str], batch_size: int = config['batch_size']) -> np.ndarray:  
        all_embeddings = []  
        for i in range(0, len(sentences), batch_size):  
            batch = sentences[i:i + batch_size]  
            inputs = self.tokenizer(batch, padding=True, truncation=True, max_length=256, return_tensors="pt")  
            inputs_on_device = {k: v.to(self.device) for k, v in inputs.items()}  
            with torch.no_grad():  
                outputs = self.model(**inputs_on_device, return_dict=True)  
            embeddings = outputs.last_hidden_state[:, 0]  
            embeddings = embeddings / embeddings.norm(dim=1, keepdim=True)  
            all_embeddings.append(embeddings.cpu().numpy())  
        return np.vstack(all_embeddings)  

# 定义向量库索引类  
class VectorStoreIndex:  
    def __init__(self, document_path: str = config['document_path'], embed_model: EmbeddingModel, chunker: TextSplitter) -> None:  
        self.documents = []  
        self.chunks = []  
        self.chunker = chunker  
        
        # 加载文档并进行切分  
        for line in open(document_path, 'r', encoding='utf-8'):  
            line = line.strip()  
            self.documents.append(line)  
            # 文本切分  
            self.chunks.extend(self.chunker.split_text(line))  
        
        self.embed_model = embed_model  
        self.vectors = self.embed_model.get_embeddings(self.chunks)  

        print(f'Loaded {len(self.documents)} documents and {len(self.chunks)} chunks from {document_path}.')  

    def get_similarity(self, vector1: List[float], vector2: List[float]) -> float:  
        dot_product = np.dot(vector1, vector2)  
        magnitude = np.linalg.norm(vector1) * np.linalg.norm(vector2)  
        if not magnitude:  
            return 0  
        return dot_product / magnitude  

    def query(self, question: str, k: int = 3) -> List[str]:  
        question_vector = self.embed_model.get_embeddings([question])[0]  
        result = np.array([self.get_similarity(question_vector, vector) for vector in self.vectors])  
        top_chunks = np.array(self.chunks)[result.argsort()[-k:][::-1]].tolist()  
        return top_chunks  

# 使用会话状态管理模型  
if 'llm' not in st.session_state:  
    st.session_state.llm = LLM()  
    print('LLM model has been initialized.')  

if 'embed_model' not in st.session_state:  
    st.session_state.embed_model = EmbeddingModel()  
    print('Embedding model has been initialized.')  

if 'index' not in st.session_state:  
    chunker = TextSplitter()  
    st.session_state.index = VectorStoreIndex(embed_model=st.session_state.embed_model, chunker=chunker)  
    print('Index has been initialized.')  
    
def should_select_school_and_faculty(question):  
    return any(keyword in question.lower() for keyword in config['keywords'])  

def filter_and_save_csv(rank_range, interest, input_file=config['input_file']):  
    # 读取输入文件  
    df = pd.read_csv(input_file)  
    
    # 解析排名范围  
    min_rank, max_rank = map(int, rank_range.split('-'))  
    
    # 过滤数据  
    filtered_df = df[(df['Rank'] >= min_rank) & (df['Rank'] <= max_rank) & (df['full_research'].str.contains(interest, case=False))]  
    
    # 创建满足条件的 CSV 数据  
    csv_result_df = filtered_df[['Institution Name', 'Rank']]  
    
    # 创建包含所有内容的 TXT 数据  
    txt_output = StringIO()  
    for _, row in filtered_df.iterrows():  
        line = (f"Ranked {row['Rank']}, {row['Institution Name']} has a faculty member "  
                f"named {row['Faculty']} who holds the title of {row['Title']}. "  
                f"They are involved in {row['Research']} research. "  
                f"More information can be found on their website: {row['Website']}. "  
                f"Contact them via email: {row['Email']}. "  
                f"Status: {row['Status']}, Chinese: {row['is_chinese']}, "  
                f"Send Priority: {row['send_priority']}, "  
                f"Full Research: {row['full_research']}, "  
                f"Unique Code: {row['Unique_Code']}.\n")  
        txt_output.write(line)  
        
    # 创建满足条件的 CSV 数据，并删除重复的 Institution Name  
    csv_result_df = csv_result_df.drop_duplicates(subset='Institution Name').reset_index(drop=True)  
    # 返回两个结果  
    return csv_result_df, txt_output.getvalue()  

    
# 页面布局  
st.set_page_config(page_title="PhD Application Assistant", layout="wide")  

# 标题和说明  
st.title("PhD Application Assistant")  
st.markdown("""  
    Hello, I am your PhD application assistant. Please select your major, and the system will recommend suitable schools and mentors for you, providing personalized advice generated based on a large language model.  
""")  

# 初始化聊天历史记录  
if "messages" not in st.session_state:  
    st.session_state["messages"] = []  

# 左侧栏用于选择研究方向  
with st.sidebar:  
    st.header("Please select your major:")  
    research_area = st.selectbox(  
    "Major:",  
    (  
        "Chemistry",  
        "Physics",  
        "Mathematics",  
        "Biochemistry",  
        "Environmental Science",  
        "Life Science",  
        "Molecular Biology & Biotechnology",  
        "Psychology",  
        "Biology",  
        "Comparative Literature",  
        "Computer Science",  
        "Decision Analytics",  
        "Civil and Environmental Engineering",  
        "Management",  
        "Ocean Science",  
        "Statistics"  
    )  
)  

# 显示历史消息  
for msg in st.session_state.messages:  
    st.chat_message(msg["role"]).write(msg["content"])  
    
# 初始化会话状态  
if 'step' not in st.session_state:  
    st.session_state.step = 0 
    
# 主输入框  
user_input = st.chat_input("Please input:", key="main_input")  

if user_input:  
    st.session_state.messages.append({"role": "user", "content": user_input})  
    st.chat_message("user").write(user_input)  

    # 判断是否需要进行学校与教授选择  
    if st.session_state.step == 0 and should_select_school_and_faculty(user_input):  
        response = 'Please input the range rank of university and your specific interest (e.g., 1-20:quantum optics).'  
        st.session_state.messages.append({"role": "assistant", "content": response})  
        st.chat_message("assistant").write(response)  
        st.session_state.step = 1  # 更新步骤  

    elif st.session_state.step == 1:  
        # 获取用户关于rank和interest的信息  
        if ";" in user_input:  
            rank_range, interest = map(str.strip, user_input.split(";"))  

            # 匹配  
            csv_result, txt_result = filter_and_save_csv(rank_range, interest)  

            # 使用临时文件  
            with tempfile.NamedTemporaryFile(delete=False, mode='w', encoding='utf-8', suffix='.txt') as temp_file:  
                temp_file.write(txt_result)  
                temp_file_path = temp_file.name  

            chunker = TextSplitter(max_chunk_size=256, overlap=50)  
            st.session_state.index = VectorStoreIndex(temp_file_path, st.session_state.embed_model, chunker)  
            print('Index has been renewed.')  

            # 构建 Markdown 格式的无序列表  
            markdown_list = "\n".join([f"- {row['Institution Name']} (Rank: {row['Rank']})" for _, row in csv_result.iterrows()])  
            response = f"Here are the university list for {interest}:\n{markdown_list}"  
            st.session_state.messages.append({"role": "assistant", "content": response})  
            st.chat_message("assistant").write(response)  

            st.session_state.step = 2  # 更新步骤  

        else:  
            st.chat_message("assistant").write("Please input the data in the format like '1-50;Physics'.")  

    elif st.session_state.step == 2:  
        # 最后使用 RAG  
        # st.session_state.messages.append({"role": "user", "content": user_input})  
        #st.chat_message("user").write(user_input)  
        context = st.session_state.index.query(user_input)  
        response = st.session_state.llm.generate(question=user_input, context=context)  
        st.session_state.messages.append({"role": "assistant", "content": response})  
        st.chat_message("assistant").write(response)  
    else:  
        # 不需选择学校和教授时，处理正常的问答  
        recommendations = st.session_state.index.query(research_area)  
        context = "\n".join(recommendations)  
        response = st.session_state.llm.generate(question=user_input, context=context)  
        st.session_state.messages.append({"role": "assistant", "content": response})  
        st.chat_message("assistant").write(response)  
