from openai import OpenAI





class GPTclient:
    def __init__(self, api_key: str, base_url: str):
        """
        初始化PumpkinAIClient类。

        :param api_key: API密钥
        :param base_url: 中转url
        """
        self.api_key = api_key
        self.base_url = base_url
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def get_response_prof_details(self, content: str) -> str:
        try:
            # 调用API
            chat_completion = self.client.chat.completions.create(
                messages=[  
                    {  
                        "role": "system",  
                        "content": "You are a helpful assistant providing detailed information about professors. Answer in a consistent style."  
                    },  
                    {  
                        "role": "user",  
                        "content": content,  
                    },  
                    {  
                        "role": "assistant",  
                        "content": (  
                            "- Website: \n"  
                            "- Email: \n"  
                            "- Full Research: \n"  
                            "- Publications:\n"  
                            "Here are 10 publications of the professor:\n"
                            "  • Publication 1\n"  
                            "  • Publication 2\n"  
                            "  • Publication 3\n"  
                            "  • Publication 4\n"  
                            "  • Publication 5\n"  
                            "  • Publication 6\n"  
                            "  • Publication 7\n"  
                            "  • Publication 8\n"  
                            "  • Publication 9\n"  
                            "  • Publication 10\n"  
                        )  
                    }  
            ] ,
                model="gpt-3.5-turbo-16k"
            )
            # 返回结果
            return chat_completion.choices[0].message.content

        except Exception as e:
            # 如果请求失败，打印错误并返回空字符串
            print(f"Error occurred: {e}")
            return ""

    def get_response_query(self, content: str) -> str:
        try:
            # 调用API
            chat_completion = self.client.chat.completions.create(
                messages=[  
                    {  
                        "role": "system",  
                        "content": "You are a versatile assistant capable of providing detailed and accurate information on a wide range of topics. Answer in a clear and consistent style."  
                    },  
                    {  
                        "role": "user",  
                        "content": content,  
                    }  
                ],
                model="gpt-3.5-turbo-16k"
            )
            # 返回结果
            return chat_completion.choices[0].message.content

        except Exception as e:
            # 如果请求失败，打印错误并返回空字符串
            print(f"Error occurred: {e}")
            return ""

    # gpt api used for keywords retrieval of user query
    def get_response_keywords(self, content: str) -> str:
        try:
            # 调用API
            chat_completion = self.client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": "You will be provided with a few sentences from the user. Your task is to extract academic keywords from the user input. The user input is their academic and personal background. The keywords should include personal details, academic focus, and relevant topics to help recommend research groups.Remember you should ignore the irrelevant message such as the region preference and the University preference. Provide only the keywords."
                    },
                    {
                        "role": "user",
                        "content": "I want to apply PhD in quantum physics, and I prefer to stay in America, I have publications in the experiment using nitrogen-vacancy centers in diamond, physics review A, 2023, I am interested in spintronics, quantum information. please recommend the group and show me their brief information, give me the reason"
                    },
                    {
                        "role": "assistant",
                        "content": "PhD in quantum physics, Nitrogen-vacancy (NV) centers in diamond, Spintronics, Quantum information"
                    },
                    {
                        "role": "user",
                        "content": content
                    }

            ],
                model="gpt-3.5-turbo-16k"
            )
            # 返回结果
            return chat_completion.choices[0].message.content

        except Exception as e:
            # 如果请求失败，打印错误并返回空字符串
            print(f"Error occurred: {e}")
            return ""

    # gpt api used for personalized recommendation
    def get_response_psnl(self, content: str) -> str:
        try:
            # 调用API
            chat_completion = self.client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": "--You will be provided with a user query about academic recommendation and a list of RAG result in our database."
                                   "--You should answer as a useful assistant to provided the accurate message(the source is limited in our database) and meet the user's requirements."
                                   "--Notice the preference of the [region/University/publication/professor title/rank/citizenship/gender] of the research group information "
                                   "--Provide the link of website if needed."
                                   "--You can use all the data in the RAG text, dilimited with the tag ###RAG##."
                                   "--Remember to think whether the RAG result totally match the user requirement."
                                   "--You can answer with bullet points to make it clearer."
                    },
                    {
                        "role": "user",
                        "content": "I want to apply PhD in quantum physics, and I prefer to stay in America, I have publications in the experiment using nitrogen-vacancy centers in diamond, physics review A, 2023, I am interested in spintronics, quantum information. please recommend the group and show me their brief information, give me the reason.   ###RAG###    Rank: 2 University: Harvard University Faculty: Kang-Kuen Ni Title: Professor Research: quantum information science Website: https://www.moore.org/investigator-detail?investigatorId=kang-kuen-ni-ph.d, https://www.physics.harvard.edu/people/facpages/ni, https://mphq.physics.harvard.edu/people/kang-kuen-ni, https://chemistry.harvard.edu/people/kang-kuen-ni, https://kni.faculty.chemistry.harvard.edu/biocv Email: nan Status: nan Is Chinese: True Send Priority: 2 Full Research: Cold atoms and molecules, ultra-cold quantum chemistry, quantum information science Publications: Article 1: Title: Precision test of statistical dynamics with state-to-state ultracold chemistry, Authors: …, MG Hu, MA Nichols, D Yang, D Xie, H Guo, KK Ni, Year: 2021, Source: https://www.nature.com/articles/s41586-021-03459-6 Article 2: Title: Nuclear spin conservation enables state-to-state control of ultracold molecular reactions, Authors: …, MA Nichols, L Zhu, G Quéméner, O Dulieu, KK Ni, Year: 2021, Source: https://www.nature.com/articles/s41557-020-00610-0 Article 3: Title: Bimolecular chemistry in the ultracold regime, Authors: Y Liu, KK Ni, Year: 2022, Source: https://www.annualreviews.org/content/journals/10.1146/annurev-physchem-090419-043244 Rank: 2 University: Harvard University Faculty: Mikhail Lukin Title: Professor Research: Quantum information processing, Quantum computers Website: https://lukin.physics.harvard.edu/people/mikhail-lukin Email: lukin@physics.harvard.edu Status: nan Is Chinese: False Send Priority: 4 Full Research: Quantum manipulation of atomic and nanoscale solid-state systems, quantum many-body physics, applications to quantum metrology and quantum information processing, realization of quantum computers and quantum networks, and development of nanoscale quantum sensors. Publications: Article 1: Title: Atom-by-atom assembly of defect-free one-dimensional cold atom arrays, Authors: …, C Senko, V Vuletic, M Greiner, MD Lukin, Year: 2016, Source: https://www.science.org/doi/abs/10.1126/science.aah3752 Article 2: Title: Efficient source of shaped single photons based on an integrated diamond nanophotonic system, Authors: …, H Park, M Lončar, MK Bhaskar, MD Lukin, Year: 2022, Source: https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.129.053603 Article 3: Title: Probing topological spin liquids on a programmable quantum simulator, Authors: …, A Vishwanath, M Greiner, V Vuletić, MD Lukin, Year: 2021, Source: https://www.science.org/doi/abs/10.1126/science.abi8794 Rank: 4 University: University of Zurich Faculty: Christian Degen Title: Professor Research: Quantum sensing Website: www.spin.ethz.ch Email: degenc@ethz.ch Status: nan Is Chinese: False Send Priority: 4 Full Research: Quantum sensing, nanoscale magnetic resonance imaging, nitrogen-vacancy centers in diamond, and single-crystal diamond nanomechanical resonators. Publications: Article 1: Title: Temperature dependence of photoluminescence intensity and spin contrast in nitrogen-vacancy centers, Authors: …, PJ Scheidegger, S Diesch, L Lorenzelli, CL Degen, Year: 2023, Source: https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.131.086903 Article 2: Title: Structural basis of NINJ1-mediated plasma membrane rupture in cell death, Authors: M Degen, JC Santos, K Pluhackova, G Cebrero…, Year: 2023, Source: https://www.nature.com/articles/s41586-023-05991-z Article 3: Title: Force-detected magnetic resonance imaging of influenza viruses in the overcoupled sensor regime, Authors: …, U Grob, H Takahashi, Y Yamauchi, CL Degen…, Year: 2022, Source: https://journals.aps.org/prapplied/abstract/10.1103/PhysRevApplied.18.034052 "
                    },
                    {
                        "role": "assistant",
                        "content": "Since you prefer to stay in America, here are the best options from the recommendations: 1. Harvard University - Kang-Kuen Ni Research Focus: Cold atoms and molecules, ultra-cold quantum chemistry, quantum information science. Relevant to You: Nitrogen-vacancy centers in diamond, quantum information science.Reason: Strong alignment with your interest in quantum information and experimental background in NV centers.Location: Cambridge, Massachusetts, USA.Contact: Kang-Kuen Ni’s profile2. Harvard University - Mikhail Lukin; Research Focus: Quantum information processing, quantum computers, quantum networks. Relevant to You: ...Both professors at Harvard University in the US offer excellent alignment with your research goals.",
                    },
                    {
                        "role": "user",
                        "content": content,
                    }
                ],
                model="gpt-3.5-turbo-16k"
            )
            # 返回结果
            return chat_completion.choices[0].message.content

        except Exception as e:
            # 如果请求失败，打印错误并返回空字符串
            print(f"Error occurred: {e}")
            return ""

    def get_response(self, content: str) -> str:
        try:
            # 调用API
            chat_completion = self.client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": content,
                    }
                ],
                model="gpt-3.5-turbo-16k"
            )
            # 返回结果
            return chat_completion.choices[0].message.content

        except Exception as e:
            # 如果请求失败，打印错误并返回空字符串
            print(f"Error occurred: {e}")
            return ""



# 使用示例
if __name__ == "__main__":
    api_key = "sk-Erm52wwWJba3F2iz620d47D7F40e4fDcB2D36e9cC22bDe09"
    base_url = "https://api.pumpkinaigc.online/v1"
    client = GPTclient(api_key, base_url)

    #demo test
    user_query = "I am applying the PhD in physics, and I'm graduated from USTC majoring in applied physics, " \
                 "my research interest is quantum mechanics in materials, I did experients in magnetic phenomena " \
                 "in photonic systems, and nanostructures, I published 2 papers in Nature Nanostructures and PRL " \
                 "about the dynamics in quantum critical points. I want to study in America and US top 10 University." \
                 " Please recommend research groups for me,and give me reasons."

    response = client.get_response(user_query)
    print(response)
