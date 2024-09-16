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

    user_content = "Hello, how are you?"
    response = client.get_response(user_content)
    print(response)
