import sys  
import json
sys.path.append("./utils") # 加入路径以便于直接运行
from gpt_api import GPTclient  
from compute_embedding import EmbeddingModel, read_text_data, save_embeddings, save_mappings
import torch  
import os  
import numpy as np  
import pandas as pd  
from sklearn.metrics.pairwise import cosine_similarity  

def compute_and_save_column_embed(  
    model_name,   
    api_key,   
    base_url,   
    csv_filename,   
    save_dir,   
    device=None  
):  
    if device is None:  
        device = 'cuda' if torch.cuda.is_available() else 'cpu'  
    
    vector_model = EmbeddingModel(  
        model_name=model_name,   
        device=device,   
        api_key=api_key,   
        base_url=base_url  
    )  
    
    df = pd.read_csv(csv_filename)  
    column_names = df.columns.tolist()  

    column_embeddings = vector_model.get_embeddings(column_names)  

    os.makedirs(save_dir, exist_ok=True)  
    save_path = os.path.join(save_dir, "column_embeddings.npy")  
    save_embeddings(np.array(column_embeddings), save_path)  

    print(f"Column embeddings saved to {save_path}")  
    
    
def get_embeddings_by_columns(  
    model_name,   
    api_key,   
    base_url,   
    csv_filename,   
    save_dir,   
    device=None  
): 
    '''
    对每一列生成embedding
    '''
    if device is None:  
        device = 'cuda' if torch.cuda.is_available() else 'cpu'  
    
    vector_model = EmbeddingModel(  
        model_name=model_name,   
        device=device,   
        api_key=api_key,   
        base_url=base_url  
    )  
    
    df = pd.read_csv(csv_filename)  
    # 存放data每一列的embedding  
    all_embeddings = []  
    # 创建保存目录 ./data/embeddings/physics，如果不存在  
    save_dir_path = os.path.join(save_dir, "physics")  
    os.makedirs(save_dir_path, exist_ok=True)  

    # 为每一列生成嵌入并保存为npy文件  
    for column in df.columns:  
        text_data = df[column].astype(str).tolist()  # 将列转换为字符串列表  
        embeddings = vector_model.get_embeddings(text_data)  # 获得嵌入  
        # 保存每一列的嵌入  
        save_path = os.path.join(save_dir_path, f"{column}_embeddings.npy")  
        np.save(save_path, embeddings)  
        
        all_embeddings.append(embeddings)  

    print(f"All column embeddings saved to {save_dir_path}")  
 

    
def main(
    model_name,   
    api_key,   
    base_url,   
    csv_filename,      
    device=None  
):  
    if device is None:  
        device = 'cuda' if torch.cuda.is_available() else 'cpu'  
    
    vector_model = EmbeddingModel(  
        model_name=model_name,   
        device=device,   
        api_key=api_key,   
        base_url=base_url  
    )    

    # User query  
    user_query = ("I am applying the PhD in physics, and I'm graduated from USTC majoring in applied physics, "  
                "my research interest is quantum mechanics in materials, I did experiments in magnetic phenomena "  
                "in photonic systems, and nanostructures, I published 2 papers in Nature Nanotechnology and PRL "  
                "about the dynamics in quantum critical points. I want to study in America and US top 10 University."  
                "Please recommend research groups for me, and give me reasons.")   
    query_embedding = vector_model.get_embeddings([user_query])  
    # print(query_embedding.shape) #(1, 768)
    
    # 计算每个列名的embedding与query_embedding的余弦相似度  
    similarities = []  
    col_embed_path = './data/embeddings/column_embeddings.npy'
    column_name_embeddings = np.load(col_embed_path)
    #print(column_embeddings.shape)
    for i, embedding in enumerate(column_name_embeddings):  
        # 计算余弦相似度  
        similarity = cosine_similarity(query_embedding.reshape(1, -1), embedding.reshape(1, -1))[0][0]  
        similarities.append(similarity)  
        #print(f"Similarity for column {i}: {similarity}")  

    
    # 归一化相似度  
    max_sim = max(similarities)  
    min_sim = min(similarities)  
    normalized_similarities = [(sim - min_sim) / (max_sim - min_sim) for sim in similarities]  
     

    # 打印所有归一化相似度  
    # print("All normalized similarities:", normalized_similarities) 
    
    json_file_path = 'data/embeddings/embeddings_files.json'  

    # 打开并读取 JSON 文件  
    with open(json_file_path, 'r') as f:  
        embeddings_files = json.load(f)   
    num_rows = None  
    scores = None  
    weights = np.array(normalized_similarities)  
    df = pd.read_csv(csv_filename)
    original_indices = df.index.tolist()  # 保存原始索引  
    #print(original_indices)
    
    for i, file_name in enumerate(embeddings_files):  
        # 构建完整文件路径  
        embeddings_dir = './data/embeddings/physics'  
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
        print("Top 3 most similar rows indices and their texts:")  
        #print(top_3_indices)
        for index in top_3_indices:  
            #print("index:", index)
            original_index = original_indices[index]  
            #print(original_index)
            print(f"Original Index: {original_index}, Score: {scores[index]}")  
            relevant_text = ", ".join(df.iloc[index].astype(str))  
            print("Text:", relevant_text)
    else:  
        print("Error: No embeddings were loaded.")  

    
    
if __name__ == '__main__':  
    model_name='BCEmbeddingmodel'
    api_key="sk-Erm52wwWJba3F2iz620d47D7F40e4fDcB2D36e9cC22bDe09" 
    base_url="https://api.pumpkinaigc.online/v1"   
    csv_filename='./data/physics_full.csv'  
    save_dir='./data/embeddings'
    '''
    compute_and_save_column_embed(  
        model_name=model_name,  
        api_key=api_key,  
        base_url=base_url,   
        csv_filename=csv_filename,  
        save_dir=save_dir  
    ) 
    get_embeddings_by_columns(  
        model_name=model_name,  
        api_key=api_key,  
        base_url=base_url,   
        csv_filename=csv_filename,  
        save_dir=save_dir    
    )  
    '''
    main(
        model_name=model_name,  
        api_key=api_key,  
        base_url=base_url,   
        csv_filename=csv_filename
    )