import os  
import pandas as pd  

# 定义文件路径  
split_dir = 'data/split'  
output_file = 'data/physics_full.csv'  

# 获取所有需要合并的文件名  
file_names = [f"updated_part_{i}.csv" for i in range(28)]  

# 初始化一个空的 DataFrame  
full_data = pd.DataFrame()  

# 遍历文件并合并  
for file_name in file_names:  
    file_path = os.path.join(split_dir, file_name)  
    if os.path.exists(file_path):  
        # 读取 CSV 文件  
        df = pd.read_csv(file_path)  
        # 追加到 full_data  
        full_data = pd.concat([full_data, df], ignore_index=True)  
    else:  
        print(f"Warning: {file_path} does not exist.")  

# 将合并后的数据保存到新的 CSV 文件  
full_data.to_csv(output_file, index=False)  
print(f"All files have been merged into {output_file}")