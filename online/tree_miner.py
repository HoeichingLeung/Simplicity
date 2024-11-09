import asyncio
import httpx


class TreeMiner:
    def __init__(self, question, max_branches=5):
        self.question = question
        self.max_branches = max_branches
        self.results = []
        self.miner_prompts = self.generate_prompts()

    def generate_prompts(self):
        """为每个miner生成不同的搜索方向的提示。"""
        prompts = [
            f"Miner 1: Begin with a focused search using the local database sources, such as professor profiles or institutional sites, to gather foundational information directly related to the question: '{self.question}'.",
            f"Miner 2: Take an alternative approach by investigating relevant social media platforms or networks. Explore sources like LinkedIn, Twitter, or academic forums to find any additional personal or professional insights for the question: '{self.question}'.",
            f"Miner 3: Expand the search scope by looking into external academic resources, including online publications, conference papers, or specialized repositories. Aim to locate any research outputs, collaborations, or citations related to the question: '{self.question}'.",
        ]
        return prompts[:self.max_branches]  # 限制最大枝条数

        async def search_with_api(self, miner_name, prompt):
            """通过web API进行搜索，获取初始结果。"""
            serper_api_key = "your_serper_api_key"  # 替换为实际API密钥
            headers = {"Authorization": f"Bearer {serper_api_key}"}
            query = {"q": prompt, "num": 3}

            async with httpx.AsyncClient() as client:
                response = await client.post("https://api.serper.dev/search", headers=headers, json=query)
                data = response.json()
                links = [result["link"] for result in data["results"]]
                return miner_name, links  # 返回链接列表以便进一步处理

        async def gpt_analyze_and_decide(self, miner_name, links):
            """GPT分析从搜索API返回的数据，并决定是否进行进一步搜索或剪枝。"""
            gpt_prompt = f"""
            You are analyzing results for {miner_name} based on the following links:
            {links}.
            Based on the question '{self.question}', analyze the relevance of these links and
            decide if further exploration in a specific direction is needed, or if the current
            information is sufficient. Provide a summary and suggest next steps.
            """

            # 用于向GPT发送请求的示例
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    "https://api.openai.com/v1/engines/gpt-4/completions",
                    headers={"Authorization": "Bearer YOUR_OPENAI_API_KEY"},
                    json={
                        "model": "gpt-4",
                        "prompt": gpt_prompt,
                        "max_tokens": 150
                    }
                )
                data = response.json()
                decision = data["choices"][0]["text"].strip()
                return miner_name, decision

        async def run_miners(self):
            """并行运行所有miner，获取初步搜索结果并进行分析。"""
            # Step 1: 通过API搜索初始数据
            search_tasks = [self.search_with_api(f"Miner {i + 1}", prompt) for i, prompt in
                            enumerate(self.miner_prompts)]
            search_results = await asyncio.gather(*search_tasks)

            # Step 2: 用GPT分析搜索结果并做出决策
            decision_tasks = [self.gpt_analyze_and_decide(miner_name, links) for miner_name, links in search_results]
            decisions = await asyncio.gather(*decision_tasks)

            # 保存分析结果
            for miner_name, decision in decisions:
                self.results.append({"miner": miner_name, "decision": decision})

            return self.results

        def consolidate_results(self):
            """整合最终结果，输出回答。"""
            final_answer = {result["miner"]: result["decision"] for result in self.results}
            return final_answer

        async def execute(self):
            """主执行流程，运行miner，分析并整合结果。"""
            await self.run_miners()
            return self.consolidate_results()

    # 示例调用
    async def main():
        professor_info = "Professor XYZ from ABC University"
        question = "What is the nationality of Professor XYZ?"

        tree_miner = TreeMiner(professor_info, question)
        final_results = await tree_miner.execute()
        print("Final Results:", final_results)

    # 运行异步主函数
    asyncio.run(main())