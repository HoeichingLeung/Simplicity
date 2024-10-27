# Simplicity
Really Simple :)

## Features Overview  

1. **Select Major:**  
   - Users can select their field of study from a dropdown menu on the left side of the application. Currently, we have opened the following majors for beta testing:  
     - Quantum Physics  
     - Artificial Intelligence (AI)  
     - Mechanical Engineering  

2. **Question Feature:**  
   - Users can ask questions about specific schools or professors to gain detailed insights and information. Our assistant provides valuable suggestions based on the latest research data and trends.  

3. **Personalized Recommendations:**  
   - Users can input personal information to receive customized advice and guidance. We utilize advanced language models to offer personalized application strategies and suggestions for each user.  

## Installation Guide  

1. Clone the repository:  
   ```bash  
   git clone https://github.com/HoeichingLeung/Simplicity.git
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the project with the following command:
   ```bash
   streamlit run src/main.py
   ```

## Structure Overview
```
Simplicity
├── model
|   └── BCEmbeddingmodel
├── config
├── data
|   ├── embeddings
|   ├── major_data
|   └── scripts
├── src
|   ├── main.py
|   └── agent_api.py
├── utils
|   ├── google-scholar-scraper
|   ├── gpt_api.py
|   ├── compute_embedding.py
|   └── compute_columns_embed.py
└── README.md
```

## Contact Information
For any inquiries, please contact hwyii0126@gmail.com.
