import chromadb
from chromadb.utils import embedding_functions
import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 创建持久化客户端
client = chromadb.PersistentClient(path="./chroma_db")

# 配置嵌入函数
openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=os.getenv("OPENAI_API_KEY"),
    api_base=os.getenv("OPENAI_BASE_URL"),  # 支持自定义端点
    model_name="text-embedding-v4"
)

# 创建集合
collection = client.get_or_create_collection(
    name="knowledge_base",
    embedding_function=openai_ef,
    metadata={"description": "我的知识库"}
)

# 添加文档
documents = [
    "人工智能是计算机科学的一个分支，致力于创建能够执行通常需要人类智能的任务的系统。",
    "机器学习是人工智能的一个子集，它使计算机能够在没有明确编程的情况下学习和改进。",
    "深度学习是机器学习的一个分支，使用多层神经网络来模拟人脑的工作方式。"
]

collection.add(
    ids=[f"doc_{i}" for i in range(len(documents))],
    documents=documents,
    metadatas=[{"topic": "AI"}, {"topic": "ML"}, {"topic": "DL"}]
)

def search_knowledge_base(query: str, n_results: int = 3):
    """在知识库中搜索相关文档"""
    results = collection.query(
        query_texts=[query],
        n_results=n_results,
        include=["documents", "metadatas", "distances"]
    )
    return results

# 测试检索
results = search_knowledge_base("什么是神经网络？")

import openai
from typing import List, Dict, Any

class RAGAgent:
    def __init__(self, collection, api_key: str, base_url: str = None):
        self.collection = collection
        self.client = openai.OpenAI(
            api_key=api_key,
            base_url=base_url
        )
    
    def retrieve(self, query: str, n_results: int = 3) -> List[Dict]:
        """检索相关文档"""
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results,
            include=["documents", "metadatas", "distances"]
        )
        
        return [{
            "content": doc,
            "metadata": meta,
            "similarity": 1 - dist
        } for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0]
        )]
    
    def generate_response(self, query: str, retrieved_docs: List[Dict]) -> str:
        """基于检索到的文档生成回答"""
        context = "\n".join([f"文档 {i+1}: {doc['content']}" 
                           for i, doc in enumerate(retrieved_docs)])
        
        prompt = f"""基于以下文档内容回答用户问题：

相关文档:
{context}

用户问题: {query}

请基于上述文档内容提供准确回答:"""

        response = self.client.chat.completions.create(
            model="deepseek-v3",
            messages=[
                {"role": "system", "content": "你是一个基于文档内容回答问题的助手。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=500
        )
        
        return response.choices[0].message.content
    
    def query(self, question: str) -> Dict[str, Any]:
        """完整的 RAG 查询流程"""
        retrieved_docs = self.retrieve(question)
        answer = self.generate_response(question, retrieved_docs)
        
        return {
            "question": question,
            "answer": answer,
            "sources": retrieved_docs
        }
    
# 创建 RAG Agent
rag_agent = RAGAgent(
    collection, 
    os.getenv("OPENAI_API_KEY"),
    os.getenv("OPENAI_BASE_URL")
)

# 进行查询
result = rag_agent.query("深度学习和机器学习有什么区别？")
print("回答:", result["answer"])