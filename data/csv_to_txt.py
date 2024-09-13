import pandas as pd  

# 读取CSV文件  
file_path = 'data/physics_full.csv'  
df = pd.read_csv(file_path)  

def format_publications(publications):  
    # 检查是否为字符串类型，避免对非字符串调用 split()  
    if isinstance(publications, str):  
        articles = publications.split(' | ')  
        formatted_articles = []  
        for i, article in enumerate(articles):  
            formatted_articles.append(f"Article {i+1}: {article}")  
        return "\n".join(formatted_articles)  
    else:  
        return "No publications available"  

def format_row(row):  
    # 将数据行格式化为文本  
    formatted_text = (  
        f"Rank: {row['Rank']}\n"  
        f"University: {row['University']}\n"  
        f"Faculty: {row['Faculty']}\n"  
        f"Title: {row['Title']}\n"  
        f"Research: {row['Research']}\n"  
        f"Website: {row['Website']}\n"  
        f"Email: {row['Email']}\n"  
        f"Status: {row['Status']}\n"  
        f"Is Chinese: {row['is_chinese']}\n"  
        f"Send Priority: {row['send_priority']}\n"  
        f"Full Research: {row['full_research']}\n"  
        f"Publications:\n{format_publications(row['publications'])}\n"  
        "###\n"  # 作为分隔符
    )  
    return formatted_text  

# 初始化输出文本  
all_text_entries = []  

# 遍历所有数据行  
for _, row in df.iterrows():  
    all_text_entries.append(format_row(row))  

# 写入文本文件  
output_file = 'data/txt_to_embed/physics_full.txt'  
with open(output_file, 'w', encoding='utf-8') as f:  
    for entry in all_text_entries:  
        f.write(entry)  

print(f"Formatted data has been written to {output_file}")