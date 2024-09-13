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

def read_text_data(file_path, delimiter='###'):  
    with open(file_path, 'r', encoding='utf-8') as file:  
        content = file.read()  
        return content.split(delimiter)  

def save_embeddings(embeddings, file_path):  
    np.save(file_path, embeddings)  

def save_mappings(mapping, file_path):  
    with open(file_path, 'w', encoding='utf-8') as f:  
        for idx, sentence in mapping.items():  
            f.write(f"{idx}\t{sentence}\n")  

def main():  
    # Initialize the embedding model class  
    model_name = 'BCEmbeddingmodel'  # Use relative path  
    device = 'cuda' if torch.cuda.is_available() else 'cpu'  
    vector_model = EmbeddingModel(model_name=model_name, device=device)  

    # Data directory and embeddings saving directory  
    data_dir = 'data/txt_to_embed'  # Relative path  
    save_dir = 'data/embeddings'  
    os.makedirs(save_dir, exist_ok=True)  

    # Iterate over txt files in the data directory, compute and save embeddings  
    for filename in os.listdir(data_dir):  
        if filename.endswith('.txt'):  
            file_path = os.path.join(data_dir, filename)  
            sentences = read_text_data(file_path)  

            # Compute embeddings and mapping  
            embeddings, index_to_sentence = vector_model.get_embeddings_with_mapping(sentences)  

            # Save embeddings  
            save_path = os.path.join(save_dir, f"{os.path.splitext(filename)[0]}_embedding.npy")  
            save_embeddings(embeddings, save_path)  
            print(f"Embedding for {filename} saved to {save_path}")  

            # Save mappings  
            mapping_path = os.path.join(save_dir, f"{os.path.splitext(filename)[0]}_mapping.txt")  
            save_mappings(index_to_sentence, mapping_path)  
            print(f"Mapping for {filename} saved to {mapping_path}")  

if __name__ == '__main__':  
    main()