import torch
import os
import logging
from transformers import AutoTokenizer, AutoModel
import yaml
import chromadb
from chromadb.config import Settings
import numpy as np

# 获取基础目录
BASE_DIR = os.getenv("pkl_base_dir", os.path.join(os.path.dirname(os.path.dirname(__file__))))
BASE_DIR = os.path.normpath(BASE_DIR).replace("\\", "/")
print(f"Base directory: {BASE_DIR}")
CONFIG_FILE = os.path.join(BASE_DIR, "embedding", "config.yaml").replace("\\", "/")

# 加载配置文件
with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
    CONFIG = yaml.safe_load(f)
    # 处理配置中的路径斜杠
    CONFIG["input_dir"] = CONFIG.get("input_dir", "").replace("\\", "/")
    CONFIG["output_dir"] = CONFIG.get("output_dir", "").replace("\\", "/")
    CONFIG["chroma_persist_dir"] = CONFIG.get("chroma_persist_dir", "").replace("\\", "/")

# 从环境变量覆盖设备配置
CONFIG['device'] = os.getenv('DEVICE', CONFIG['device'])

class ChromaEmbeddingProcessor:
    def __init__(self):
        self.logger = self._setup_logger()
        self.tokenizer = AutoTokenizer.from_pretrained(CONFIG["model_name"])
        self.model = AutoModel.from_pretrained(CONFIG["model_name"]).to(CONFIG["device"])
        self._create_chroma_client()
        self._create_output_dir()
        
    def _setup_logger(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)
        
    def _create_chroma_client(self):
        # 初始化Chroma客户端
        self.chroma_client = chromadb.Client(Settings(
            persist_directory=CONFIG.get("chroma_persist_dir", os.path.join(BASE_DIR, "chroma_data"))
        ))
        self.logger.info("Chroma client initialized")
    
    def _create_output_dir(self):
        # 硬编码输出目录名称为output_chroma
        output_dir = os.path.join(BASE_DIR, "output_chroma")
        os.makedirs(output_dir.replace("\\", "/"), exist_ok=True)
        self.logger.info(f"Output directory created: {output_dir}")
    
    def _read_batch(self, filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip()]
            
    def generate_embeddings(self):
        input_path = CONFIG["input_dir"]
        for txt_file in os.listdir(input_path):
            if txt_file.endswith(".txt"):
                file_path = os.path.join(input_path, txt_file).replace("\\", "/")
                self.logger.info(f"Processing file: {file_path}")
                
                texts = self._read_batch(file_path)
                embeddings = []
                
                # 处理每个batch并立即存储到Chroma
                collection_name = f"{os.path.splitext(txt_file)[0]}_chroma"
                
                for i in range(0, len(texts), CONFIG["batch_size"]):
                    batch = texts[i:i+CONFIG["batch_size"]]
                    # 合并batch中的所有文本生成一个综合文本（使用换行符连接）
                    combined_text = "\n".join(batch)
                    inputs = self.tokenizer(
                        combined_text,
                        padding=True,
                        truncation=True,
                        max_length=CONFIG["max_length"],
                        return_tensors='pt'
                    ).to(CONFIG["device"])
                    
                    with torch.no_grad():
                        outputs = self.model(**inputs)
                        batch_embedding = outputs.last_hidden_state[:, 0].cpu().numpy()
                        if CONFIG["normalize"]:
                            batch_embedding = batch_embedding / np.linalg.norm(batch_embedding, axis=1, keepdims=True)
                        # 立即存储当前batch的综合文本和embedding
                        self._save_to_chroma(
                            collection_name, 
                            [combined_text], 
                            [batch_embedding.tolist()[0]],
                            batch_id=i//CONFIG["batch_size"]
                        )
    
    def _save_to_chroma(self, collection_name, documents, embeddings, batch_id=0):
        # 创建或获取集合
        collection = self.chroma_client.get_or_create_collection(name=collection_name)
        
        # 添加文档和嵌入（使用唯一的ID）
        id = f"id_{batch_id}"
        collection.add(
            embeddings=embeddings,
            documents=documents,
            ids=[id]
        )
        self.logger.info(f"Saved {len(documents)} embedding(s) to Chroma collection '{collection_name}' with ID: {id}")

if __name__ == "__main__":
    processor = ChromaEmbeddingProcessor()
    processor.generate_embeddings()
