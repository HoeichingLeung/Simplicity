import os  
import numpy as np  
from transformers import AutoTokenizer, AutoModel  
import torch  
from typing import List  

class EmbeddingModel:  
    def __init__(self, model_name: str, device: str = 'cuda'):  
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)  
        self.model = AutoModel.from_pretrained(model_name)  
        self.device = device  
        self.model.to(self.device)  
    
    def get_embeddings(self, sentences: List[str], batch_size: int = 4) -> np.ndarray:  
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

def read_text_data(file_path):  
    with open(file_path, 'r', encoding='utf-8') as file:  
        return file.readlines()  # 获取每一行文本数据作为独立的句子  

def save_embeddings(embeddings, file_path):  
    np.save(file_path, embeddings)  

def main():  
    # 实例化向量模型类  
    model_name = '../model/BCEmbeddingmodel'  # 使用相对路径  
    device = 'cuda' if torch.cuda.is_available() else 'cpu'  
    vector_model = EmbeddingModel(model_name=model_name, device=device)  

    # 数据目录和嵌入保存目录  
    data_dir = '../data/txt_to_embed'  # 相对路径  
    save_dir = '../data/embeddings'  
    os.makedirs(save_dir, exist_ok=True)  

    # 遍历data目录中的txt文件，计算并保存嵌入  
    for filename in os.listdir(data_dir):  
        if filename.endswith('.txt'):  
            file_path = os.path.join(data_dir, filename)  
            sentences = read_text_data(file_path)  

            # 计算嵌入  
            embeddings = vector_model.get_embeddings(sentences)  

            # 保存嵌入  
            save_path = os.path.join(save_dir, f"{os.path.splitext(filename)[0]}_embedding.npy")  
            save_embeddings(embeddings, save_path)  
            print(f"Embedding for {filename} saved to {save_path}")  

if __name__ == '__main__':  
    main()