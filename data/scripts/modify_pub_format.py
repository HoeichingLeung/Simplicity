import pandas as pd  

# 读取CSV文件  
file_path = 'D:/Simplicity/data/physics_full.csv'  
df = pd.read_csv(file_path)  

# 移除publications列中的换行符  
df['publications'] = df['publications'].apply(lambda x: x.replace('\n', ' ') if isinstance(x, str) else x)  

# 保存修改后的CSV文件  
df.to_csv(file_path, index=False)  

print(f"Newlines in 'publications' column have been removed and saved to {file_path}")