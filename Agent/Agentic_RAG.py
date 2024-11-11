import os
from typing import List
import faiss
import numpy as np
from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent, AgentOutputParser
from langchain.prompts import StringPromptTemplate
import re

# 设置 OpenAI API 密钥
os.environ["OPENAI_API_KEY"] = 'your-openai-api-key'

# 1. 数据准备
# 示例文档
documents = [
    {"title": "量子计算简介", "content": "量子计算利用量子力学现象，如叠加和纠缠，进行信息处理。"},
    {"title": "自然语言处理", "content": "自然语言处理是人工智能的一个分支，旨在让计算机理解和生成人类语言。"},
    {"title": "机器学习基础", "content": "机器学习是一种通过数据训练模型，以进行预测和决策的技术。"},
    {"title": "深度学习概念", "content": "深度学习是机器学习的一个子领域，主要使用神经网络进行数据建模。"},
    {"title": "向量搜索原理", "content": "向量搜索通过将文本转换为向量，在高维空间中进行相似性匹配，以检索相关信息。"}
]

# 初始化嵌入模型
embeddings = OpenAIEmbeddings()

# 提取文档内容用于嵌入
texts = [doc["content"] for doc in documents]

# 生成嵌入向量
embedding_vectors = embeddings.embed_documents(texts)

# 创建FAISS向量索引
dimension = len(embedding_vectors[0])
faiss_index = faiss.IndexFlatL2(dimension)
faiss_index.add(np.array(embedding_vectors).astype('float32'))

# 使用FAISS创建VectorStore
vector_store = FAISS(embedding_vectors, faiss_index, texts)

# 2. 定义检索工具
def search_tool(query: str) -> str:
    """
    使用向量搜索引擎检索与查询最相关的文档。
    """
    results = vector_store.similarity_search(query, k=2)
    response = ""
    for idx, doc in enumerate(results):
        response += f"结果 {idx+1}:\n标题: {documents[idx]['title']}\n内容: {doc.page_content}\n\n"
    return response

# 定义检索工具为 LangChain Tool
search = Tool(
    name = "向量搜索工具",
    func=search_tool,
    description="用于检索与查询相关的文档。"
)

# 3. 定义生成工具
def generate_response_tool(query: str, context: str) -> str:
    """
    使用大型语言模型生成基于上下文的回答。
    """
    llm = OpenAI(model="text-davinci-003", temperature=0.2)
    prompt = f"根据以下上下文信息，为以下查询生成一个简明且准确的回答。\n\n上下文:\n{context}\n\n查询:\n{query}\n\n回答:"
    response = llm(prompt)
    return response.strip()

# 定义生成工具为 LangChain Tool
generate_response = Tool(
    name = "生成回答工具",
    func=generate_response_tool,
    description="根据上下文信息生成回答。"
)

# 4. 定义 Agent
# 自定义解析器，用于解析 LLM 的输出并决定下一步行动
class CustomOutputParser(AgentOutputParser):
    def parse(self, llm_output: str) -> str:
        # 定义一个简单的解析逻辑，提取生成的回答
        regex = r"回答:\s*(.*)"
        match = re.search(regex, llm_output, re.DOTALL)
        if match:
            return match.group(1).strip()
        return llm_output.strip()

# 定义提示模板
prompt_template = StringPromptTemplate(
    input_variables=["input", "tools"],
    template=(
        "你是一个智能助理，能够高效地检索和生成信息。\n"
        "以下是可用的工具:\n"
        "{tools}\n\n"
        "根据用户的查询，选择合适的工具来检索信息，并生成回答。\n"
        "用户查询: {input}\n\n"
        "你的回答:"
    )
)

# 创建 Agent
tools = [search, generate_response]
agent_chain = LLMSingleActionAgent(
    llm=OpenAI(model="text-davinci-003"),
    output_parser=CustomOutputParser(),
    stop=["\nObservation:"],
    allowed_tools=[tool.name for tool in tools]
)

# 创建 Agent Executor
agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent_chain,
    tools=tools,
    verbose=True
)

# 5. 执行查询
if __name__ == "__main__":
    user_query = "请解释一下量子计算的基本原理。"
    print(f"用户查询: {user_query}\n")
    response = agent_executor.run(user_query)
    print(f"生成的回答:\n{response}")