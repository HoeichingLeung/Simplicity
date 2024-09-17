import os
import sys
import numpy as np
sys.path.append("./utils")
from gpt_api import GPTclient
from transformers import AutoTokenizer, AutoModel
import torch
from typing import List
import pandas as pd
from langchain_community.utilities import GoogleSerperAPIWrapper


def web_search(prof_name: str):
    os.environ["SERPER_API_KEY"] = "88a8892a02409063f02a3bb97ac08b36fb213ae7"
    search = GoogleSerperAPIWrapper()
    search_result = prof_name + ":"
    search_item = prof_name + "information"
    search_result += str(search.run(search_item)) + '\n'
    return search_result
    # results = search.results(search_item)
    # pprint.pp(results)


class EmbeddingModel(GPTclient):
    def __init__(self, api_key: str, base_url: str, model_name: str, device: str = 'cuda'):
        """
        :param api_key: API密钥
        :param base_url: 中转url
        :param csv_file_path: CSV文件路径
        """
        super().__init__(api_key, base_url)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.device = device
        self.model.to(self.device)
        self.vector = np.empty((1,768))
        
    def get_embeddings(self, sentences: List[str], batch_size: int = 8) -> np.ndarray:
        all_embeddings = []
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i:i + batch_size]
            #print(batch)
            inputs = self.tokenizer(batch, padding=True, truncation=True, max_length=256, return_tensors="pt")
            #print(inputs)
            inputs_on_device = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self.model(**inputs_on_device, return_dict=True)
            embeddings = outputs.last_hidden_state[:, 0]
            embeddings = embeddings / embeddings.norm(dim=1, keepdim=True)
            all_embeddings.append(embeddings.cpu().numpy())
        return np.vstack(all_embeddings)

    def get_embeddings_with_mapping(self, sentences: List[str], batch_size: int = 4):
        all_embeddings = []
        index_to_sentence = {}

        for i in range(0, len(sentences), batch_size):
            batch = sentences[i:i + batch_size]

            # Create a mapping of index to sentence
            for idx, sentence in enumerate(batch, start=i):
                index_to_sentence[idx] = sentence

            inputs = self.tokenizer(batch, padding=True, truncation=True, max_length=256, return_tensors="pt")
            inputs_on_device = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self.model(**inputs_on_device, return_dict=True)
            embeddings = outputs.last_hidden_state[:, 0]
            embeddings = embeddings / embeddings.norm(dim=1, keepdim=True)
            all_embeddings.append(embeddings.cpu().numpy())
        return np.vstack(all_embeddings), index_to_sentence
    def get_similarity(self, vector1: List[float], vector2: List[float]) -> float:
        """
        calculate cosine similarity between two vectors
        """
        dot_product = np.dot(vector1, vector2)
        magnitude = np.linalg.norm(vector1) * np.linalg.norm(vector2)
        if not magnitude:
            return 0
        return dot_product / magnitude
    def load_embed(self, file_path):
        array_data = np.load(file_path)
        self.vectors = array_data
        print('>load_embed...')
        print(f'The embed size is{array_data.shape}')


    def query_for_sentences(self, question: str, k: int) -> str:
        question_vector = self.get_embeddings([question])[0]  
        result = np.array([self.get_similarity(question_vector, vector) for vector in self.vectors])  
        top_k_indices = result.argsort()[-k:][::-1]  
        df = pd.read_csv('data/embeddings/physics_full_mapping.csv')  
        # Retrieve the sentences corresponding to indices  
        matched_sentences = df[df['Index'].isin(top_k_indices)]['Sentence'].tolist()  
        # Join the list of sentences into a single string  
        result_string = ' '.join(matched_sentences)

        # 增加网络搜索faculty信息
        faculty_names = [entry.split('Faculty: ')[1].split(' Title:')[0] for entry in matched_sentences]
        #print(">>>>>>faculty name:", faculty_names)
        web_result = []
        for name in faculty_names:
            web_result.append(web_search(name))

        #print(">web_result:", web_result)

        return matched_sentences, web_result



def read_text_data(file_path, delimiter='###'):  
    with open(file_path, 'r', encoding='utf-8') as file:  
        content = file.read()  
        return content.split(delimiter)  

def save_embeddings(embeddings, file_path):  
    np.save(file_path, embeddings)  

# def save_mappings(mapping, file_path):
#     with open(file_path, 'w', encoding='utf-8') as f:
#         for idx, sentence in mapping.items():
#             f.write(f"{idx}\t{sentence}\n")
def save_mappings(mapping, file_path):
    df = pd.DataFrame(list(mapping.items()), columns=['Index', 'Sentence'])
    df['Sentence'] = df['Sentence'].replace('\n', ' ', regex=True)
    df.set_index('Index', inplace=True)
    df.to_csv(file_path, encoding='utf-8')

def main():
    # 实例化向量模型类
    model_name = 'BCEmbeddingmodel'  # 使用相对路径
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    vector_model = EmbeddingModel(model_name=model_name, device=device)

    # 数据目录和嵌入保存目录
    data_dir = './data/txt_to_embed'  # 相对路径
    save_dir = './data/embeddings'
    os.makedirs(save_dir, exist_ok=True)

    # 遍历data目录中的txt文件，计算并保存嵌入
    for filename in os.listdir(data_dir):
        if filename.endswith('.txt'):
            file_path = os.path.join(data_dir, filename)
            sentences = read_text_data(file_path)

            # 计算嵌入与映射
            embeddings, index_to_sentence = vector_model.get_embeddings_with_mapping(sentences)

            # 保存嵌入
            save_path = os.path.join(save_dir, f"{os.path.splitext(filename)[0]}_embedding.npy")
            save_embeddings(embeddings, save_path)
            print(f"Embedding for {filename} saved to {save_path}")

            # 保存映射
            mapping_path = os.path.join(save_dir, f"{os.path.splitext(filename)[0]}_mapping.csv")
            save_mappings(index_to_sentence, mapping_path)
            print(f"Mapping for {filename} saved to {mapping_path}")

    '''
    testing RAG
    '''
    #file_path_npy = './data/embeddings/physics_full_embedding.npy'
    #test_model = EmbeddingModel(model_name=model_name, device=device)
    #test_model.load_embed(file_path = file_path_npy)
    #rag_result = test_model.query_for_sentences(k=3,question = "I am interested in first-principles exploration of novel quantum physics and materials, focusing on emergent quantum phenomena" )
    #print(">rag:",rag_result)


if __name__ == '__main__':
    main()