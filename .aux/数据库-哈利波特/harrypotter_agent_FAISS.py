import torch
import os
import yaml
import json  # 添加json模块用于元数据处理
from transformers import AutoTokenizer, AutoModel
import faiss
import numpy as np

# 获取基础目录
BASE_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__))).replace("\\", "/")
#print(f"Base directory: {BASE_DIR}")
CONFIG_FILE = os.path.join(BASE_DIR, "embedding", "config.yaml").replace("\\", "/")

# 加载配置文件
with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
    CONFIG = yaml.safe_load(f)
    CONFIG["output_dir"] = CONFIG.get("output_dir", "").replace("\\", "/")

class HarryPotterAgent:
    def __init__(self):
        self.device = CONFIG.get("device", "cpu")
        self.model_name = CONFIG.get("model_name", "bert-base-uncased")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name).to(self.device)
        self.index = self._load_index()
    
    def _load_index(self):
        """加载所有batch_*.index文件并合并为全局索引"""
        import shutil
        index_dir = CONFIG["output_dir"]
        temp_dir = CONFIG["temp_dir"]
        
        # 创建temp_dir目录
        os.makedirs(temp_dir, exist_ok=True)
        
        # 获取所有batch_*.index文件
        index_files = [f for f in os.listdir(index_dir) if f.startswith("batch_") and f.endswith(".index")]
        if not index_files:
            raise FileNotFoundError(f"未找到batch索引文件: {index_dir}")
        
        # 按文件名中的序号排序（batch_0.index, batch_1.index...）
        index_files.sort(key=lambda x: int(x.split("_")[1].split(".")[0]))
        
        # 将索引文件复制到temp_dir
        temp_files = []
        for file_name in index_files:
            src_path = os.path.join(index_dir, file_name)
            dst_path = os.path.join(temp_dir, file_name)
            shutil.copy2(src_path, dst_path)
            temp_files.append(dst_path)
        
        # 初始化全局索引和batch信息
        self.global_index = None
        self.batch_info = []
        current_id = 0
        
        # 逐个读取并合并
        for file_index, file_name in enumerate(index_files):
            # 使用temp_dir路径避免中文字符问题
            temp_file_path = os.path.join(temp_dir, file_name)
            index = faiss.read_index(temp_file_path)
            
            # 获取当前索引的向量
            vectors = index.reconstruct_n(0, index.ntotal)
            if self.global_index is None:
                self.global_index = faiss.IndexFlatL2(index.d)
            self.global_index.add(vectors)
            
            # 记录batch信息
            self.batch_info.append({
                "start_id": current_id,
                "end_id": current_id + index.ntotal - 1,
                "file_name": file_name,
                "file_index": file_index
            })
            
            current_id += index.ntotal
        
        # 删除temp_dir中的临时文件
        import shutil
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        
        return self.global_index
    
    def _generate_embedding(self, text):
        """生成输入文本的嵌入向量"""
        inputs = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=CONFIG.get("max_length", 512),
            return_tensors='pt'
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            embedding = outputs.last_hidden_state[:, 0].cpu().numpy()
            if CONFIG.get("normalize", True):
                embedding = embedding / np.linalg.norm(embedding, axis=1, keepdims=True)
        
        return embedding
    
    def get_top_k_files(self, question, k=3):
        """
        根据问题获取最相关的k个文件路径
        
        参数:
            question (str): 用户的问题
            k (int): 需要返回的最相关文件数量
            
        返回:
            list: 包含匹配文件路径的列表，格式为[文件路径1, 文件路径2, 文件路径3]
        """
        # 生成问题的嵌入向量
        question_embedding = self._generate_embedding(question)
        
        # 执行Faiss搜索
        distances, indices = self.index.search(question_embedding, k)
        
        # 解析结果并生成文件路径列表
        file_paths = []
        for idx in indices[0]:
            if idx < len(self.batch_info):
                file_name = self.batch_info[idx]["file_name"]
                file_path = os.path.join(CONFIG["output_dir"], file_name).replace("\\", "/")
                file_paths.append(file_path)
            else:
                file_paths.append("unknown")
                
        # 返回前k个文件路径（按距离排序）
        return file_paths[:k]
    
    def find_relevant_batches(self, question, top_batches=3):
        """
        查找与问题最相关的批次
        
        参数:
            question (str): 用户的问题
            top_batches (int): 需要返回的最相关批次数量
            
        返回:
            list: 包含匹配批次信息的列表，格式为[(文件路径, 距离)]
        """
        # 生成问题的嵌入向量
        question_embedding = self._generate_embedding(question)
        
        # 执行Faiss搜索
        distances, indices = self.index.search(question_embedding, top_batches)
        
        # 解析结果（使用batch_info中的文件名信息）
        results = []
        for i, idx in enumerate(indices[0]):
            # 使用batch_info中的文件名信息构建路径
            # 确保idx在有效范围内
            idx = idx % len(self.batch_info) if len(self.batch_info) > 0 else 0
            
            if idx < len(self.batch_info):
                file_name = self.batch_info[idx]["file_name"]
                # 将索引文件路径替换为txt_batches中的txt文件
                txt_file = file_name.replace('.index', '.txt')
                file_path = os.path.join(BASE_DIR, "txt_batches", txt_file).replace("\\", "/")
            else:
                # 默认使用第一个batch的信息
                file_name = self.batch_info[0]["file_name"]
                txt_file = file_name.replace('.index', '.txt')
                file_path = os.path.join(BASE_DIR, "txt_batches", txt_file).replace("\\", "/")
                
            # 只保留文件路径和距离
            results.append((file_path, distances[0][i]))
        
        # 按距离从小到大排序
        results.sort(key=lambda x: x[1])
        return results
    
    def _load_metadata(self):
        """加载批次元数据文件（批次索引到文件路径的映射）"""
        metadata_path = os.path.join(CONFIG["output_dir"], "batch_metadata.json").replace("\\", "/")
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"批次元数据文件未找到: {metadata_path}")
        
        with open(metadata_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    

if __name__ == "__main__":
    agent = HarryPotterAgent()
    question = "总结一下哈利波特学习漂浮咒的过程"
    # 测试get_top_k_files方法
    top_files = agent.get_top_k_files(question, 3)
    print(f"与问题最相关的前{len(top_files)}个文件:")
    for i, file_path in enumerate(top_files, 1):
        print(f"{i}. 文件: {file_path}")
    
    # 测试find_relevant_batches方法
    batch_results = agent.find_relevant_batches(question)
    print("\n完整批次信息:")
    #print(batch_results)
    for i, (file_path, distance) in enumerate(batch_results, 1):
        print(f"{i}. 文件: {file_path}, 距离: {distance:.4f}")


def question_to_context(question, top_batches=3):
    """将问题转换为上下文文本列表"""
    agent = HarryPotterAgent()  # 创建HarryPotterAgent实例
    batch_results = agent.find_relevant_batches(question, top_batches)  # 调用新方法
    file_paths = [result[0] for result in batch_results]  # 提取文件路径
    contexts = []
    for path in file_paths:
        with open(path, 'r', encoding='utf-8-sig', errors='ignore') as f:
            contexts.append(f.read())
    return contexts

if __name__ == "__main__":
    # 示例使用
    sample_question = "哈利波特的父亲是谁"
    results = question_to_context(sample_question)
    
    # 输出前50个字符
    for i, text in enumerate(results):
        print(f"段落{i+1}前50字符: \n{text[:50]}\n")
